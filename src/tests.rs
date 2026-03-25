use crate::{
    poseidon2::{
        BabyBearPoseidon2Backend, BabyBearPoseidon2_16HashAir, KoalaBearPoseidon2Backend,
        KoalaBearPoseidon2_16HashAir, POSEIDON2_16_WIDTH,
    },
    relation, HashInvocationAir, RelationChallenge, RelationField, StarkRelationBackend,
};
use alloc::vec::Vec;
use p3_air::SymbolicExpressionExt;
use p3_field::{Algebra, BasedVectorSpace, Field, PrimeCharacteristicRing};
use spongefish::{Permutation, Unit};
use spongefish_circuit::{
    allocator::FieldVar,
    permutation::{LinearEquation, PermutationInstanceBuilder, PermutationWitnessBuilder},
};
use spongefish_poseidon2::{BabyBearPoseidon2_16, KoalaBearPoseidon2_16};

const TEST_LINEAR_WIDTH: usize = 1;

fn sample_input<F, const WIDTH: usize>() -> [F; WIDTH]
where
    F: Field + Unit + PartialEq,
{
    core::array::from_fn(|i| F::from_usize(i + 1))
}

fn build_relation_instance_and_witness<B, P, const WIDTH: usize, const LIN_WIDTH: usize>(
    permutation: P,
    input: [RelationField<B>; WIDTH],
    public_outputs: &[(usize, RelationField<B>)],
) -> (
    PermutationInstanceBuilder<RelationField<B>, WIDTH>,
    PermutationWitnessBuilder<P, WIDTH>,
)
where
    B: StarkRelationBackend,
    P: Permutation<WIDTH, U = RelationField<B>>,
    RelationField<B>: Field + Unit + PartialEq + Send + Sync,
{
    let instance = PermutationInstanceBuilder::<RelationField<B>, WIDTH>::new();
    let witness = PermutationWitnessBuilder::<P, WIDTH>::new(permutation);

    let input_vars = instance.allocator().allocate_public::<WIDTH>(&input);
    let output_vars = instance.allocate_permutation(&input_vars);
    let output_vals = witness.allocate_permutation(&input);

    instance.allocator().set_public_vars(
        public_outputs.iter().map(|(idx, _)| output_vars[*idx]),
        public_outputs.iter().map(|(_, val)| *val),
    );

    instance.add_equation(LinearEquation::new(
        core::iter::once((
            <RelationField<B> as PrimeCharacteristicRing>::ONE,
            output_vars[0],
        ))
        .chain((1..LIN_WIDTH).map(|_| {
            (
                <RelationField<B> as PrimeCharacteristicRing>::ZERO,
                FieldVar(0),
            )
        })),
        output_vals[0],
    ));
    witness.add_equation(LinearEquation::new(
        core::iter::once((
            <RelationField<B> as PrimeCharacteristicRing>::ONE,
            output_vals[0],
        ))
        .chain((1..LIN_WIDTH).map(|_| {
            (
                <RelationField<B> as PrimeCharacteristicRing>::ZERO,
                <RelationField<B> as PrimeCharacteristicRing>::ZERO,
            )
        })),
        output_vals[0],
    ));

    (instance, witness)
}

fn run_hash_relation_checks<B, H, P, const WIDTH: usize, const LIN_WIDTH: usize>(
    backend: &B,
    hash: &H,
    permutation: P,
) where
    B: StarkRelationBackend,
    H: HashInvocationAir<RelationField<B>, WIDTH> + Sync,
    P: Permutation<WIDTH, U = RelationField<B>> + Clone,
    RelationField<B>: Field + Unit + PartialEq + Send + Sync,
    RelationChallenge<B>: BasedVectorSpace<RelationField<B>>,
    SymbolicExpressionExt<RelationField<B>, RelationChallenge<B>>: Algebra<RelationChallenge<B>>,
{
    let input = sample_input::<RelationField<B>, WIDTH>();
    let expected_output = permutation.permute(&input);
    let public_outputs = Vec::from([
        (1usize, expected_output[1]),
        (2usize, expected_output[2]),
        (3usize, expected_output[3]),
    ]);

    let (instance, witness) = build_relation_instance_and_witness::<B, P, WIDTH, LIN_WIDTH>(
        permutation.clone(),
        input,
        &public_outputs,
    );

    let proof = relation::prove::<B, H, P, WIDTH, LIN_WIDTH>(backend, hash, &instance, &witness);
    assert!(relation::verify::<B, H, WIDTH, LIN_WIDTH>(backend, hash, &instance, &proof).is_ok());

    let mut bad_proof = proof.clone();
    bad_proof[0] ^= 0x01;
    assert!(
        relation::verify::<B, H, WIDTH, LIN_WIDTH>(backend, hash, &instance, &bad_proof).is_err()
    );

    let bad_public_outputs = Vec::from([
        (
            1usize,
            expected_output[1] + <RelationField<B> as PrimeCharacteristicRing>::ONE,
        ),
        (2usize, expected_output[2]),
        (3usize, expected_output[3]),
    ]);
    let (bad_instance, _) = build_relation_instance_and_witness::<B, P, WIDTH, LIN_WIDTH>(
        permutation,
        input,
        &bad_public_outputs,
    );
    assert!(
        relation::verify::<B, H, WIDTH, LIN_WIDTH>(backend, hash, &bad_instance, &proof).is_err()
    );

    let mut shifted_proof = proof;
    shifted_proof.insert(0, 0x00);
    assert!(
        relation::verify::<B, H, WIDTH, LIN_WIDTH>(backend, hash, &instance, &shifted_proof)
            .is_err()
    );
}

#[test]
fn poseidon2_16_relation_proof_and_false_checks() {
    run_hash_relation_checks::<
        BabyBearPoseidon2Backend,
        BabyBearPoseidon2_16HashAir,
        BabyBearPoseidon2_16,
        POSEIDON2_16_WIDTH,
        TEST_LINEAR_WIDTH,
    >(
        &BabyBearPoseidon2Backend::default(),
        &BabyBearPoseidon2_16HashAir::default(),
        BabyBearPoseidon2_16::default(),
    );

    run_hash_relation_checks::<
        KoalaBearPoseidon2Backend,
        KoalaBearPoseidon2_16HashAir,
        KoalaBearPoseidon2_16,
        POSEIDON2_16_WIDTH,
        TEST_LINEAR_WIDTH,
    >(
        &KoalaBearPoseidon2Backend::default(),
        &KoalaBearPoseidon2_16HashAir::default(),
        KoalaBearPoseidon2_16::default(),
    );
}
