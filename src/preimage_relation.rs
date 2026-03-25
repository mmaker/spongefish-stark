use alloc::{vec, vec::Vec};

use p3_field::Field;
use p3_matrix::Matrix;
use p3_uni_stark::{
    prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed, StarkGenericConfig,
};
use spongefish::{Permutation, Unit, VerificationError, VerificationResult};
use spongefish_circuit::{
    allocator::FieldVar,
    permutation::{PermutationInstanceBuilder, PermutationWitnessBuilder},
};

use crate::{HashInvocationAir, HashPreimageAir, QueryAnswerPair, RelationField, StarkRelationBackend};

/// A single hash-preimage statement with some output lanes marked public.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreimageStatement<F, const WIDTH: usize> {
    pub input: [F; WIDTH],
    pub output: [F; WIDTH],
    pub public_outputs: [Option<F>; WIDTH],
}

/// Build an instance and witness for a batch of preimage statements.
pub fn build_instance_and_witness<B, P, const WIDTH: usize>(
    permutation: P,
    statements: impl IntoIterator<Item = PreimageStatement<RelationField<B>, WIDTH>>,
) -> (
    PermutationInstanceBuilder<RelationField<B>, WIDTH>,
    PermutationWitnessBuilder<P, WIDTH>,
)
where
    B: StarkRelationBackend,
    P: Permutation<WIDTH, U = RelationField<B>> + Clone,
    RelationField<B>: Field + Unit + PartialEq + Send + Sync,
{
    let instance = PermutationInstanceBuilder::<RelationField<B>, WIDTH>::new();
    let witness = PermutationWitnessBuilder::<P, WIDTH>::new(permutation.clone());

    for statement in statements {
        let input_vars = instance.allocator().allocate_vars::<WIDTH>();
        let output_vars = instance.allocator().allocate_vars::<WIDTH>();
        instance.add_permutation(input_vars, output_vars);
        let witness_output = permutation.permute(&statement.input);
        assert_eq!(
            witness_output, statement.output,
            "preimage statement output does not match the witness permutation"
        );
        witness.add_permutation(&statement.input, &statement.output);

        let public_vars = statement
            .public_outputs
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| value.map(|value| (output_vars[idx], value)))
            .collect::<Vec<_>>();
        instance.allocator().set_public_vars(
            public_vars.iter().map(|(var, _)| *var),
            public_vars.iter().map(|(_, value)| *value),
        );
    }

    (instance, witness)
}

pub fn prove<B, H, P, const WIDTH: usize>(
    backend: &B,
    hash: &H,
    instance: &PermutationInstanceBuilder<RelationField<B>, WIDTH>,
    witness: &PermutationWitnessBuilder<P, WIDTH>,
) -> Vec<u8>
where
    B: StarkRelationBackend,
    H: HashInvocationAir<RelationField<B>, WIDTH> + Sync,
    P: Permutation<WIDTH, U = RelationField<B>>,
    RelationField<B>: Field + Unit + PartialEq + Send + Sync,
{
    let statements = collect_statements::<B, P, WIDTH>(instance, witness)
        .expect("preimage relation only supports fresh permutations with public outputs only");
    let public_outputs = statements
        .iter()
        .map(|statement| statement.public_outputs)
        .collect::<Vec<_>>();
    let air = HashPreimageAir::new(hash.clone(), public_outputs);
    let witness_rows = statements
        .iter()
        .map(|statement| QueryAnswerPair::new(statement.input, statement.output))
        .collect::<Vec<_>>();
    let trace = air.build_trace_rows(&witness_rows, 2);
    assert!(
        trace.height().is_power_of_two(),
        "trace height must be a power of two"
    );
    let degree_bits = trace.height().trailing_zeros() as usize;

    let config = backend.config(2024);
    let preprocessed = setup_preprocessed(&config, &air, degree_bits);
    let proof = prove_with_preprocessed(
        &config,
        &air,
        trace,
        &[],
        preprocessed.as_ref().map(|(pp, _)| pp),
    );
    postcard::to_allocvec(&proof).expect("proof serialization should succeed")
}

pub fn verify<B, H, const WIDTH: usize>(
    backend: &B,
    hash: &H,
    instance: &PermutationInstanceBuilder<RelationField<B>, WIDTH>,
    proof_bytes: &[u8],
) -> VerificationResult<()>
where
    B: StarkRelationBackend,
    H: HashInvocationAir<RelationField<B>, WIDTH> + Sync,
    RelationField<B>: Field + Unit + PartialEq + Send + Sync,
{
    let proof: p3_uni_stark::Proof<B::Config> =
        postcard::from_bytes(proof_bytes).map_err(|_| VerificationError)?;
    let public_outputs = collect_public_outputs(instance).ok_or(VerificationError)?;
    let config = backend.config(2024);
    let air = HashPreimageAir::new(hash.clone(), public_outputs);
    let degree_bits = proof.degree_bits.saturating_sub(config.is_zk());
    let preprocessed = setup_preprocessed(&config, &air, degree_bits);
    verify_with_preprocessed(
        &config,
        &air,
        &proof,
        &[],
        preprocessed.as_ref().map(|(_, vk)| vk),
    )
    .map_err(|_| VerificationError)
}

fn collect_statements<B, P, const WIDTH: usize>(
    instance: &PermutationInstanceBuilder<RelationField<B>, WIDTH>,
    witness: &PermutationWitnessBuilder<P, WIDTH>,
) -> Option<Vec<PreimageStatement<RelationField<B>, WIDTH>>>
where
    B: StarkRelationBackend,
    P: Permutation<WIDTH, U = RelationField<B>>,
    RelationField<B>: Field + Unit + PartialEq + Send + Sync,
{
    if !instance.linear_constraints().as_ref().is_empty()
        || !witness.linear_constraints().as_ref().is_empty()
    {
        return None;
    }

    let constraints = instance.constraints();
    let witness_rows = witness.trace();
    if constraints.as_ref().len() != witness_rows.as_ref().len() {
        return None;
    }

    let public_outputs = collect_public_outputs(instance)?;
    Some(
        constraints
            .as_ref()
            .iter()
            .zip(witness_rows.as_ref().iter())
            .zip(public_outputs)
            .map(|((_vars, values), public_outputs)| PreimageStatement {
                input: values.input,
                output: values.output,
                public_outputs,
            })
            .collect(),
    )
}

fn collect_public_outputs<F, const WIDTH: usize>(
    instance: &PermutationInstanceBuilder<F, WIDTH>,
) -> Option<Vec<[Option<F>; WIDTH]>>
where
    F: Field + Unit + PartialEq + Clone,
{
    if !instance.linear_constraints().as_ref().is_empty() {
        return None;
    }

    let constraints = instance.constraints();
    let public_vars = instance.public_vars();
    let max_var = constraints
        .as_ref()
        .iter()
        .flat_map(|pair| pair.input.iter().chain(pair.output.iter()))
        .chain(public_vars.iter().map(|(var, _)| var))
        .map(|var| var.0)
        .max()
        .unwrap_or(0);
    let mut public_values = vec![None; max_var + 1];
    let mut output_owners = vec![None; max_var + 1];
    let mut input_vars = vec![false; max_var + 1];

    for (statement_idx, pair) in constraints.as_ref().iter().enumerate() {
        for var in pair.input {
            input_vars[var.0] = true;
        }
        for var in pair.output {
            if output_owners[var.0].replace(statement_idx).is_some() {
                return None;
            }
        }
    }

    for (var, value) in public_vars {
        if var == FieldVar::ZERO {
            continue;
        }
        if input_vars.get(var.0).copied().unwrap_or(false) {
            return None;
        }
        if output_owners.get(var.0).and_then(|owner| *owner).is_none() {
            return None;
        }
        if public_values[var.0].replace(value).is_some() {
            return None;
        }
    }

    Some(
        constraints
            .as_ref()
            .iter()
            .map(|pair| core::array::from_fn(|idx| public_values[pair.output[idx].0].clone()))
            .collect(),
    )
}
