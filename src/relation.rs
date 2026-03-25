use alloc::{boxed::Box, string::ToString, vec, vec::Vec};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicExpressionExt, WindowAccess,
};
use p3_batch_stark::{BatchProof, ProverData, StarkInstance};
use p3_field::{Algebra, BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use p3_uni_stark::StarkGenericConfig;
use spongefish::{Permutation, Unit, VerificationError, VerificationResult};
use spongefish_circuit::{
    allocator::FieldVar,
    permutation::{LinearConstraints, PermutationInstanceBuilder, PermutationWitnessBuilder},
};

use crate::{
    HashInvocationAir, QueryAnswerPair, RelationChallenge, RelationField, StarkRelationBackend,
};

/// The first lookup, checking that outputs re-appear as inputs.
pub const IO_LOOKUP_NAME: &str = "in-out";
/// The value-level lookup emitted by the real hash AIR.
pub const HASH_VALUES_LOOKUP_NAME: &str = "hash-values";
/// The value-level lookup routed from unique hash rows to logical bindings.
pub const BINDING_VALUES_LOOKUP_NAME: &str = "binding-values";
/// The second lookup, checking that public variables are correctly assigned.
pub const PUB_LOOKUP_NAME: &str = "public-vars";
/// The third lookup, checking linear relations between inputs and outputs.
pub const LIN_LOOKUP_NAME: &str = "linear-constraints";

type LinearConstraintsInstance<F> = LinearConstraints<FieldVar, F>;
type LinearConstraintsWitness<F> = LinearConstraints<F, F>;

#[repr(C)]
struct LookupCols<T, const WIDTH: usize> {
    input_vars: [T; WIDTH],
    output_vars: [T; WIDTH],
    output_multiplicities: [T; WIDTH],
    input_multiplicities: [T; WIDTH],
    input_public: [T; WIDTH],
    output_public: [T; WIDTH],
    input_linear_constraints: [T; WIDTH],
    output_linear_constraints: [T; WIDTH],
}

#[repr(C)]
struct BindingCols<T, const WIDTH: usize> {
    input: [T; WIDTH],
    output: [T; WIDTH],
    active: T,
}

#[repr(C)]
struct ValueMultiplicityCols<T, const WIDTH: usize> {
    input: [T; WIDTH],
    output: [T; WIDTH],
    multiplicity: T,
    active: T,
}

#[repr(C)]
struct PublicLookupCols<T> {
    var: T,
    val: T,
    multiplicity: T,
}

#[repr(C)]
struct LinearConstraintCols<T, const LIN_WIDTH: usize> {
    linear_coefficients: [T; LIN_WIDTH],
    linear_combination: [T; LIN_WIDTH],
    image: T,
}

#[repr(C)]
struct LinearConstraintPreprocessedCols<T, const LIN_WIDTH: usize> {
    linear_vars: [T; LIN_WIDTH],
    image_value: T,
    linear_multiplicities: [T; LIN_WIDTH],
}

impl<T, const WIDTH: usize> Borrow<LookupCols<T, WIDTH>> for [T] {
    fn borrow(&self) -> &LookupCols<T, WIDTH> {
        let (prefix, shorts, suffix) = unsafe { self.align_to::<LookupCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize> BorrowMut<LookupCols<T, WIDTH>> for [T] {
    fn borrow_mut(&mut self) -> &mut LookupCols<T, WIDTH> {
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<LookupCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<T> Borrow<PublicLookupCols<T>> for [T] {
    fn borrow(&self) -> &PublicLookupCols<T> {
        let (prefix, shorts, suffix) = unsafe { self.align_to::<PublicLookupCols<T>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize> Borrow<BindingCols<T, WIDTH>> for [T] {
    fn borrow(&self) -> &BindingCols<T, WIDTH> {
        let (prefix, shorts, suffix) = unsafe { self.align_to::<BindingCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize> BorrowMut<BindingCols<T, WIDTH>> for [T] {
    fn borrow_mut(&mut self) -> &mut BindingCols<T, WIDTH> {
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<BindingCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<T, const WIDTH: usize> Borrow<ValueMultiplicityCols<T, WIDTH>> for [T] {
    fn borrow(&self) -> &ValueMultiplicityCols<T, WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to::<ValueMultiplicityCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize> BorrowMut<ValueMultiplicityCols<T, WIDTH>> for [T] {
    fn borrow_mut(&mut self) -> &mut ValueMultiplicityCols<T, WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to_mut::<ValueMultiplicityCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<T> BorrowMut<PublicLookupCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut PublicLookupCols<T> {
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<PublicLookupCols<T>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<T, const LIN_WIDTH: usize> Borrow<LinearConstraintCols<T, LIN_WIDTH>> for [T] {
    fn borrow(&self) -> &LinearConstraintCols<T, LIN_WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to::<LinearConstraintCols<T, LIN_WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const LIN_WIDTH: usize> BorrowMut<LinearConstraintCols<T, LIN_WIDTH>> for [T] {
    fn borrow_mut(&mut self) -> &mut LinearConstraintCols<T, LIN_WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to_mut::<LinearConstraintCols<T, LIN_WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<T, const LIN_WIDTH: usize> Borrow<LinearConstraintPreprocessedCols<T, LIN_WIDTH>> for [T] {
    fn borrow(&self) -> &LinearConstraintPreprocessedCols<T, LIN_WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to::<LinearConstraintPreprocessedCols<T, LIN_WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const LIN_WIDTH: usize> BorrowMut<LinearConstraintPreprocessedCols<T, LIN_WIDTH>>
    for [T]
{
    fn borrow_mut(&mut self) -> &mut LinearConstraintPreprocessedCols<T, LIN_WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to_mut::<LinearConstraintPreprocessedCols<T, LIN_WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

const fn num_lookup_cols<const WIDTH: usize>() -> usize {
    size_of::<LookupCols<u8, WIDTH>>()
}

const fn num_public_lookup_cols() -> usize {
    size_of::<PublicLookupCols<u8>>()
}

const fn num_binding_main_cols<const WIDTH: usize>() -> usize {
    size_of::<BindingCols<u8, WIDTH>>()
}

const fn num_value_main_cols<const WIDTH: usize>() -> usize {
    size_of::<ValueMultiplicityCols<u8, WIDTH>>()
}

const fn num_linear_main_cols<const LIN_WIDTH: usize>() -> usize {
    size_of::<LinearConstraintCols<u8, LIN_WIDTH>>()
}

const fn num_linear_preprocessed_cols<const LIN_WIDTH: usize>() -> usize {
    size_of::<LinearConstraintPreprocessedCols<u8, LIN_WIDTH>>()
}

#[derive(Clone)]
struct HashLookupAir<H, const WIDTH: usize> {
    hash: H,
}

#[derive(Clone)]
struct ValueMultiplicityAir<const WIDTH: usize>;

#[derive(Clone)]
struct BindingLookupAir<F, const WIDTH: usize, const LIN_WIDTH: usize> {
    instance: PermutationInstanceBuilder<F, WIDTH>,
    linear_constraints: Option<LinearConstraintsInstance<F>>,
    trace_len: usize,
}

#[derive(Clone)]
struct PublicVarLookupAir<F, const WIDTH: usize> {
    instance: PermutationInstanceBuilder<F, WIDTH>,
    trace_len: usize,
}

#[derive(Clone)]
struct LinearConstraintsAir<F, const WIDTH: usize, const LIN_WIDTH: usize> {
    instance: PermutationInstanceBuilder<F, WIDTH>,
    constraints: LinearConstraintsInstance<F>,
    trace_len: usize,
}

#[derive(Clone)]
enum GenericHashRelationAir<H, F, const WIDTH: usize, const LIN_WIDTH: usize> {
    Hash(Box<HashLookupAir<H, WIDTH>>),
    Unique(ValueMultiplicityAir<WIDTH>),
    Binding(BindingLookupAir<F, WIDTH, LIN_WIDTH>),
    Public(PublicVarLookupAir<F, WIDTH>),
    Linear(LinearConstraintsAir<F, WIDTH, LIN_WIDTH>),
}

impl<H, const WIDTH: usize> HashLookupAir<H, WIDTH> {
    fn new(hash: H) -> Self {
        Self { hash }
    }
}

impl<const WIDTH: usize> ValueMultiplicityAir<WIDTH> {
    fn new() -> Self {
        Self
    }
}

impl<F, const WIDTH: usize, const LIN_WIDTH: usize> BindingLookupAir<F, WIDTH, LIN_WIDTH> {
    fn new_with_linear(
        instance: PermutationInstanceBuilder<F, WIDTH>,
        linear_constraints: LinearConstraintsInstance<F>,
        trace_len: usize,
    ) -> Self {
        Self {
            instance,
            linear_constraints: Some(linear_constraints),
            trace_len,
        }
    }
}

impl<F, const WIDTH: usize> PublicVarLookupAir<F, WIDTH> {
    fn new(instance: PermutationInstanceBuilder<F, WIDTH>, trace_len: usize) -> Self {
        assert!(trace_len.is_power_of_two());
        Self {
            instance,
            trace_len,
        }
    }
}

impl<F, const WIDTH: usize, const LIN_WIDTH: usize> LinearConstraintsAir<F, WIDTH, LIN_WIDTH> {
    fn new(
        instance: PermutationInstanceBuilder<F, WIDTH>,
        constraints: LinearConstraintsInstance<F>,
        trace_len: usize,
    ) -> Self {
        assert!(trace_len.is_power_of_two());
        assert!(constraints.as_ref().len() <= trace_len);
        Self {
            instance,
            constraints,
            trace_len,
        }
    }
}

pub fn prove<B, H, P, const WIDTH: usize, const LIN_WIDTH: usize>(
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
    RelationChallenge<B>: BasedVectorSpace<RelationField<B>>,
    SymbolicExpressionExt<RelationField<B>, RelationChallenge<B>>: Algebra<RelationChallenge<B>>,
{
    let (mut airs, traces) =
        generate_trace_rows_with_linear::<H, RelationField<B>, P, WIDTH, LIN_WIDTH>(
            hash, instance, witness,
        );
    let log_degrees = trace_degree_bits(&traces);
    let config = backend.config(2024);
    let log_ext_degrees = log_ext_degrees(&log_degrees, &config);
    let prover_data = ProverData::from_airs_and_degrees(&config, &mut airs, &log_ext_degrees);
    let common = &prover_data.common;
    let publics = vec![Vec::new(); airs.len()];
    let trace_refs = traces.iter().collect::<Vec<_>>();
    let instances = StarkInstance::new_multiple(&airs, &trace_refs, &publics, common);
    let proof = p3_batch_stark::prove_batch(&config, &instances, &prover_data);
    postcard::to_allocvec(&proof).expect("proof serialization should succeed")
}

pub fn verify<B, H, const WIDTH: usize, const LIN_WIDTH: usize>(
    backend: &B,
    hash: &H,
    instance: &PermutationInstanceBuilder<RelationField<B>, WIDTH>,
    proof_bytes: &[u8],
) -> VerificationResult<()>
where
    B: StarkRelationBackend,
    H: HashInvocationAir<RelationField<B>, WIDTH> + Sync,
    RelationField<B>: Field + Unit + PartialEq + Send + Sync,
    RelationChallenge<B>: BasedVectorSpace<RelationField<B>>,
    SymbolicExpressionExt<RelationField<B>, RelationChallenge<B>>: Algebra<RelationChallenge<B>>,
{
    let linear_constraints = instance.linear_constraints();
    validate_linear_constraints::<LIN_WIDTH, _, _>(&linear_constraints, "instance");
    let config = backend.config(2024);
    let proof: BatchProof<B::Config> =
        postcard::from_bytes(proof_bytes).map_err(|_| VerificationError)?;
    let degree_bits = &proof.degree_bits;
    assert_eq!(degree_bits.len(), 5);
    let trace_len = |idx: usize| 1usize << degree_bits[idx].saturating_sub(config.is_zk());
    let mut airs = vec![
        GenericHashRelationAir::Hash(Box::new(HashLookupAir::<H, WIDTH>::new(hash.clone()))),
        GenericHashRelationAir::Unique(ValueMultiplicityAir::new()),
        GenericHashRelationAir::Binding(
            BindingLookupAir::<RelationField<B>, WIDTH, LIN_WIDTH>::new_with_linear(
                instance.clone(),
                linear_constraints.clone(),
                trace_len(2),
            ),
        ),
        GenericHashRelationAir::Public(PublicVarLookupAir::new(instance.clone(), trace_len(3))),
        GenericHashRelationAir::Linear(
            LinearConstraintsAir::<RelationField<B>, WIDTH, LIN_WIDTH>::new(
                instance.clone(),
                linear_constraints,
                trace_len(4),
            ),
        ),
    ];
    let prover_data = ProverData::from_airs_and_degrees(&config, &mut airs, &proof.degree_bits);
    let publics = vec![Vec::new(); airs.len()];
    p3_batch_stark::verify_batch(&config, &airs, &proof, &publics, &prover_data.common)
        .map_err(|_| VerificationError)
}

fn generate_trace_rows_with_linear<H, F, P, const WIDTH: usize, const LIN_WIDTH: usize>(
    hash: &H,
    instance: &PermutationInstanceBuilder<F, WIDTH>,
    witness: &PermutationWitnessBuilder<P, WIDTH>,
) -> (
    Vec<GenericHashRelationAir<H, F, WIDTH, LIN_WIDTH>>,
    Vec<DenseMatrix<F>>,
)
where
    H: HashInvocationAir<F, WIDTH> + Sync,
    P: Permutation<WIDTH, U = F>,
    F: Field + Unit + PartialEq + Send + Sync,
{
    let linear_constraints = instance.linear_constraints();
    let linear_witness = witness.linear_constraints();
    let witness_trace = witness.trace();
    let witness_rows = witness_trace.as_ref();
    let instance_constraints = instance.constraints();
    assert_eq!(
        linear_constraints.as_ref().len(),
        linear_witness.as_ref().len()
    );
    assert_eq!(instance_constraints.as_ref().len(), witness_rows.len());
    validate_linear_constraints::<LIN_WIDTH, _, _>(&linear_constraints, "instance");
    validate_linear_constraints::<LIN_WIDTH, _, _>(&linear_witness, "witness");

    let compressed_witness = compress_query_answer_pairs(witness_rows);
    let hash_trace = build_hash_lookup_trace(hash, &compressed_witness);
    let unique_trace = build_value_multiplicity_trace(&compressed_witness);
    let binding_trace = build_binding_lookup_trace(witness_rows);
    let public_trace = build_public_lookup_main_trace::<F>(
        instance.public_vars().len().next_power_of_two().max(1),
    );
    let linear_trace = build_linear_constraints_trace::<F, LIN_WIDTH>(&linear_witness);

    let airs = vec![
        GenericHashRelationAir::Hash(Box::new(HashLookupAir::<H, WIDTH>::new(hash.clone()))),
        GenericHashRelationAir::Unique(ValueMultiplicityAir::new()),
        GenericHashRelationAir::Binding(BindingLookupAir::<F, WIDTH, LIN_WIDTH>::new_with_linear(
            instance.clone(),
            linear_constraints.clone(),
            binding_trace.height(),
        )),
        GenericHashRelationAir::Public(PublicVarLookupAir::new(
            instance.clone(),
            public_trace.height(),
        )),
        GenericHashRelationAir::Linear(LinearConstraintsAir::<F, WIDTH, LIN_WIDTH>::new(
            instance.clone(),
            linear_constraints,
            linear_trace.height(),
        )),
    ];

    (
        airs,
        vec![
            hash_trace,
            unique_trace,
            binding_trace,
            public_trace,
            linear_trace,
        ],
    )
}

impl<H, F, const WIDTH: usize, const LIN_WIDTH: usize> BaseAir<F>
    for GenericHashRelationAir<H, F, WIDTH, LIN_WIDTH>
where
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn width(&self) -> usize {
        match self {
            Self::Hash(air) => <HashLookupAir<H, WIDTH> as BaseAir<F>>::width(air),
            Self::Unique(air) => <ValueMultiplicityAir<WIDTH> as BaseAir<F>>::width(air),
            Self::Binding(air) => <BindingLookupAir<F, WIDTH, LIN_WIDTH> as BaseAir<F>>::width(air),
            Self::Public(air) => <PublicVarLookupAir<F, WIDTH> as BaseAir<F>>::width(air),
            Self::Linear(air) => {
                <LinearConstraintsAir<F, WIDTH, LIN_WIDTH> as BaseAir<F>>::width(air)
            }
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::Hash(air) => <HashLookupAir<H, WIDTH> as BaseAir<F>>::preprocessed_trace(air),
            Self::Unique(air) => {
                <ValueMultiplicityAir<WIDTH> as BaseAir<F>>::preprocessed_trace(air)
            }
            Self::Binding(air) => {
                <BindingLookupAir<F, WIDTH, LIN_WIDTH> as BaseAir<F>>::preprocessed_trace(air)
            }
            Self::Public(air) => {
                <PublicVarLookupAir<F, WIDTH> as BaseAir<F>>::preprocessed_trace(air)
            }
            Self::Linear(air) => {
                <LinearConstraintsAir<F, WIDTH, LIN_WIDTH> as BaseAir<F>>::preprocessed_trace(air)
            }
        }
    }
}

impl<H, F, const WIDTH: usize> BaseAir<F> for HashLookupAir<H, WIDTH>
where
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn width(&self) -> usize {
        self.hash.main_width()
    }
}

impl<F, const WIDTH: usize> BaseAir<F> for ValueMultiplicityAir<WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn width(&self) -> usize {
        num_value_main_cols::<WIDTH>()
    }
}

impl<F, const WIDTH: usize, const LIN_WIDTH: usize> BaseAir<F>
    for BindingLookupAir<F, WIDTH, LIN_WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn width(&self) -> usize {
        num_binding_main_cols::<WIDTH>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let trace = build_binding_preprocessed_trace::<F, WIDTH>(
            &self.instance,
            self.linear_constraints.as_ref(),
        );
        Some(pad_dense_matrix_to_height(trace, self.trace_len))
    }
}

impl<F, const WIDTH: usize> BaseAir<F> for PublicVarLookupAir<F, WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let trace = build_public_lookup_table_trace(&self.instance);
        Some(pad_dense_matrix_to_height(trace, self.trace_len))
    }
}

impl<F, const WIDTH: usize, const LIN_WIDTH: usize> BaseAir<F>
    for LinearConstraintsAir<F, WIDTH, LIN_WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn width(&self) -> usize {
        num_linear_main_cols::<LIN_WIDTH>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let trace =
            build_lin_lookup_trace::<F, WIDTH, LIN_WIDTH>(&self.instance, &self.constraints);
        Some(pad_dense_matrix_to_height(trace, self.trace_len))
    }
}

impl<AB, H, F, const WIDTH: usize, const LIN_WIDTH: usize> Air<AB>
    for GenericHashRelationAir<H, F, WIDTH, LIN_WIDTH>
where
    AB: AirBuilder<F = F>,
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Hash(air) => <HashLookupAir<H, WIDTH> as Air<AB>>::eval(air, builder),
            Self::Unique(air) => <ValueMultiplicityAir<WIDTH> as Air<AB>>::eval(air, builder),
            Self::Binding(air) => {
                <BindingLookupAir<F, WIDTH, LIN_WIDTH> as Air<AB>>::eval(air, builder)
            }
            Self::Public(air) => <PublicVarLookupAir<F, WIDTH> as Air<AB>>::eval(air, builder),
            Self::Linear(air) => {
                <LinearConstraintsAir<F, WIDTH, LIN_WIDTH> as Air<AB>>::eval(air, builder)
            }
        }
    }
}

impl<AB, H, F, const WIDTH: usize> Air<AB> for HashLookupAir<H, WIDTH>
where
    AB: AirBuilder<F = F>,
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn eval(&self, builder: &mut AB) {
        self.hash.eval(builder);
    }
}

impl<AB, F, const WIDTH: usize> Air<AB> for ValueMultiplicityAir<WIDTH>
where
    AB: AirBuilder<F = F>,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<AB, F, const WIDTH: usize, const LIN_WIDTH: usize> Air<AB>
    for BindingLookupAir<F, WIDTH, LIN_WIDTH>
where
    AB: AirBuilder<F = F>,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<AB, F, const WIDTH: usize> Air<AB> for PublicVarLookupAir<F, WIDTH>
where
    AB: AirBuilder<F = F>,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<AB, F, const WIDTH: usize, const LIN_WIDTH: usize> Air<AB>
    for LinearConstraintsAir<F, WIDTH, LIN_WIDTH>
where
    AB: AirBuilder<F = F>,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &LinearConstraintCols<_, LIN_WIDTH> = main.current_slice().borrow();
        let preprocessed = builder.preprocessed();
        let prep: &LinearConstraintPreprocessedCols<_, LIN_WIDTH> =
            preprocessed.current_slice().borrow();
        let image_value = prep.image_value;
        let mut sum = AB::Expr::ZERO;
        for i in 0..LIN_WIDTH {
            sum += local.linear_coefficients[i] * local.linear_combination[i];
        }
        builder.assert_eq(sum, local.image);
        builder.assert_eq(local.image, image_value);
    }
}

impl<H, F, const WIDTH: usize, const LIN_WIDTH: usize> LookupAir<F>
    for GenericHashRelationAir<H, F, WIDTH, LIN_WIDTH>
where
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        match self {
            Self::Hash(air) => <HashLookupAir<H, WIDTH> as LookupAir<F>>::add_lookup_columns(air),
            Self::Unique(air) => {
                <ValueMultiplicityAir<WIDTH> as LookupAir<F>>::add_lookup_columns(air)
            }
            Self::Binding(air) => {
                <BindingLookupAir<F, WIDTH, LIN_WIDTH> as LookupAir<F>>::add_lookup_columns(air)
            }
            Self::Public(air) => {
                <PublicVarLookupAir<F, WIDTH> as LookupAir<F>>::add_lookup_columns(air)
            }
            Self::Linear(air) => {
                <LinearConstraintsAir<F, WIDTH, LIN_WIDTH> as LookupAir<F>>::add_lookup_columns(air)
            }
        }
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        match self {
            Self::Hash(air) => <HashLookupAir<H, WIDTH> as LookupAir<F>>::get_lookups(air),
            Self::Unique(air) => <ValueMultiplicityAir<WIDTH> as LookupAir<F>>::get_lookups(air),
            Self::Binding(air) => {
                <BindingLookupAir<F, WIDTH, LIN_WIDTH> as LookupAir<F>>::get_lookups(air)
            }
            Self::Public(air) => <PublicVarLookupAir<F, WIDTH> as LookupAir<F>>::get_lookups(air),
            Self::Linear(air) => {
                <LinearConstraintsAir<F, WIDTH, LIN_WIDTH> as LookupAir<F>>::get_lookups(air)
            }
        }
    }
}

impl<H, F, const WIDTH: usize> LookupAir<F> for HashLookupAir<H, WIDTH>
where
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        let symbolic = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let main = symbolic.main();
        let row = main.row_slice(0).expect("symbolic row should exist");
        let frame = self.hash.row_frame(&row);
        let invocation = self.hash.invocation::<SymbolicAirBuilder<F>>(&frame);
        let selector = self.hash.lookup_selector::<SymbolicAirBuilder<F>>(&frame);
        type Expr<F> = <SymbolicAirBuilder<F> as AirBuilder>::Expr;

        vec![Lookup::new(
            Kind::Global(HASH_VALUES_LOOKUP_NAME.to_string()),
            vec![value_lookup_elements::<_, Expr<F>, WIDTH>(
                &invocation.input,
                &invocation.output,
            )],
            vec![Direction::Send.multiplicity(selector)],
            vec![0],
        )]
    }
}

impl<F, const WIDTH: usize> LookupAir<F> for ValueMultiplicityAir<WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        let symbolic = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let main = symbolic.main();
        let row = main.row_slice(0).expect("symbolic row should exist");
        let cols: &ValueMultiplicityCols<_, WIDTH> = (*row).borrow();
        type Expr<F> = <SymbolicAirBuilder<F> as AirBuilder>::Expr;
        let values: Vec<Expr<F>> =
            value_lookup_elements::<_, Expr<F>, WIDTH>(&cols.input, &cols.output);
        let active: Expr<F> = cols.active.into();
        let multiplicity: Expr<F> = cols.multiplicity.into();

        vec![
            Lookup::new(
                Kind::Global(HASH_VALUES_LOOKUP_NAME.to_string()),
                vec![values.clone()],
                vec![Direction::Receive.multiplicity(active.clone())],
                vec![0],
            ),
            Lookup::new(
                Kind::Global(BINDING_VALUES_LOOKUP_NAME.to_string()),
                vec![values],
                vec![Direction::Send.multiplicity(active * multiplicity)],
                vec![1],
            ),
        ]
    }
}

impl<F, const WIDTH: usize, const LIN_WIDTH: usize> LookupAir<F>
    for BindingLookupAir<F, WIDTH, LIN_WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        let symbolic = SymbolicAirBuilder::<F>::new(AirLayout {
            preprocessed_width: num_lookup_cols::<WIDTH>(),
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let main = symbolic.main();
        let row = main.row_slice(0).expect("symbolic row should exist");
        let binding: &BindingCols<_, WIDTH> = (*row).borrow();
        let preprocessed = symbolic.preprocessed();
        let lookup_row = preprocessed
            .row_slice(0)
            .expect("symbolic row should exist");
        let lookup_column: &LookupCols<_, WIDTH> = (*lookup_row).borrow();
        type Expr<F> = <SymbolicAirBuilder<F> as AirBuilder>::Expr;
        let active: Expr<F> = binding.active.into();

        let mut lookups = vec![Lookup::new(
            Kind::Global(BINDING_VALUES_LOOKUP_NAME.to_string()),
            vec![value_lookup_elements::<_, Expr<F>, WIDTH>(
                &binding.input,
                &binding.output,
            )],
            vec![Direction::Receive.multiplicity(active.clone())],
            vec![0],
        )];

        for i in 0..WIDTH {
            let input = binding.input[i].clone();
            let output = binding.output[i].clone();
            let input_var = lookup_column.input_vars[i];
            let output_var = lookup_column.output_vars[i];
            let input_multiplicity: Expr<F> = lookup_column.input_multiplicities[i].into();
            let output_multiplicity: Expr<F> = lookup_column.output_multiplicities[i].into();
            let input_public: Expr<F> = lookup_column.input_public[i].into();
            let output_public: Expr<F> = lookup_column.output_public[i].into();
            let input_linear: Expr<F> = lookup_column.input_linear_constraints[i].into();
            let output_linear: Expr<F> = lookup_column.output_linear_constraints[i].into();

            lookups.push(Lookup::new(
                Kind::Global(IO_LOOKUP_NAME.to_string()),
                vec![
                    vec![input_var.into(), input.clone().into()],
                    vec![output_var.into(), output.clone().into()],
                ],
                vec![
                    Direction::Send.multiplicity(active.clone() * input_multiplicity),
                    Direction::Receive.multiplicity(active.clone() * output_multiplicity),
                ],
                vec![3 * i + 1],
            ));
            lookups.push(Lookup::new(
                Kind::Global(PUB_LOOKUP_NAME.to_string()),
                vec![
                    vec![output_var.into(), output.clone().into()],
                    vec![input_var.into(), input.clone().into()],
                ],
                vec![
                    Direction::Send.multiplicity(active.clone() * output_public),
                    Direction::Send.multiplicity(active.clone() * input_public),
                ],
                vec![3 * i + 2],
            ));
            lookups.push(Lookup::new(
                Kind::Global(LIN_LOOKUP_NAME.to_string()),
                vec![
                    vec![input_var.into(), input.into()],
                    vec![output_var.into(), output.into()],
                ],
                vec![
                    Direction::Send.multiplicity(active.clone() * input_linear),
                    Direction::Send.multiplicity(active.clone() * output_linear),
                ],
                vec![3 * i + 3],
            ));
        }
        lookups
    }
}

impl<F, const WIDTH: usize> LookupAir<F> for PublicVarLookupAir<F, WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        let symbolic = SymbolicAirBuilder::<F>::new(AirLayout {
            preprocessed_width: num_public_lookup_cols(),
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let preprocessed = symbolic.preprocessed();
        let row = preprocessed
            .row_slice(0)
            .expect("symbolic preprocessed row should exist");
        let public_column: &PublicLookupCols<_> = (*row).borrow();
        vec![Lookup::new(
            Kind::Global(PUB_LOOKUP_NAME.to_string()),
            vec![vec![public_column.var.into(), public_column.val.into()]],
            vec![Direction::Receive.multiplicity(public_column.multiplicity.into())],
            vec![0],
        )]
    }
}

impl<F, const WIDTH: usize, const LIN_WIDTH: usize> LookupAir<F>
    for LinearConstraintsAir<F, WIDTH, LIN_WIDTH>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        let symbolic = SymbolicAirBuilder::<F>::new(AirLayout {
            preprocessed_width: num_linear_preprocessed_cols::<LIN_WIDTH>(),
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let main = symbolic.main();
        let row = main.row_slice(0).expect("symbolic row should exist");
        let main_cols: &LinearConstraintCols<_, LIN_WIDTH> = (*row).borrow();
        let preprocessed = symbolic.preprocessed();
        let pre_row = preprocessed
            .row_slice(0)
            .expect("symbolic row should exist");
        let pre_cols: &LinearConstraintPreprocessedCols<_, LIN_WIDTH> = (*pre_row).borrow();

        let mut entries = Vec::with_capacity(LIN_WIDTH);
        for i in 0..LIN_WIDTH {
            entries.push((
                vec![
                    pre_cols.linear_vars[i].into(),
                    main_cols.linear_combination[i].into(),
                ],
                pre_cols.linear_multiplicities[i].into(),
                Direction::Receive,
            ));
        }
        let (element_exprs, multiplicities_exprs): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .map(|(elements, multiplicity, direction)| {
                (elements, direction.multiplicity(multiplicity))
            })
            .unzip();
        vec![Lookup::new(
            Kind::Global(LIN_LOOKUP_NAME.to_string()),
            element_exprs,
            multiplicities_exprs,
            vec![0],
        )]
    }
}

#[derive(Clone)]
struct CountedQueryAnswerPair<T, const WIDTH: usize> {
    pair: QueryAnswerPair<T, WIDTH>,
    multiplicity: usize,
}

fn build_hash_lookup_trace<H, F, const WIDTH: usize>(
    hash: &H,
    witness_rows: &[CountedQueryAnswerPair<F, WIDTH>],
) -> DenseMatrix<F>
where
    H: HashInvocationAir<F, WIDTH>,
    F: Field,
{
    hash.build_trace(
        &witness_rows
            .iter()
            .map(|row| row.pair.clone())
            .collect::<Vec<_>>(),
        2,
    )
}

fn build_value_multiplicity_trace<F, const WIDTH: usize>(
    witness_rows: &[CountedQueryAnswerPair<F, WIDTH>],
) -> DenseMatrix<F>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    let width = num_value_main_cols::<WIDTH>();
    let height = witness_rows.len().next_power_of_two().max(1);
    let mut values = vec![<F as PrimeCharacteristicRing>::ZERO; width * height];

    for (row_idx, row_data) in witness_rows.iter().enumerate() {
        let offset = row_idx * width;
        let row = &mut values[offset..offset + width];
        let cols: &mut ValueMultiplicityCols<F, WIDTH> = row.borrow_mut();
        cols.input = row_data.pair.input;
        cols.output = row_data.pair.output;
        cols.multiplicity = F::from_usize(row_data.multiplicity);
        cols.active = <F as PrimeCharacteristicRing>::ONE;
    }

    DenseMatrix::new(values, width)
}

fn build_binding_lookup_trace<F, const WIDTH: usize>(
    witness_rows: &[QueryAnswerPair<F, WIDTH>],
) -> DenseMatrix<F>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    let width = num_binding_main_cols::<WIDTH>();
    let height = witness_rows.len().next_power_of_two().max(1);
    let mut values = vec![<F as PrimeCharacteristicRing>::ZERO; width * height];

    for (row_idx, pair) in witness_rows.iter().enumerate() {
        let offset = row_idx * width;
        let row = &mut values[offset..offset + width];
        let cols: &mut BindingCols<F, WIDTH> = row.borrow_mut();
        cols.input = pair.input;
        cols.output = pair.output;
        cols.active = <F as PrimeCharacteristicRing>::ONE;
    }

    DenseMatrix::new(values, width)
}

fn build_binding_preprocessed_trace<F, const WIDTH: usize>(
    instance: &PermutationInstanceBuilder<F, WIDTH>,
    linear_constraints: Option<&LinearConstraintsInstance<F>>,
) -> DenseMatrix<F>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    let input_outputs = instance.constraints();
    let vars_count = instance.allocator().vars_count();
    let output_count = input_outputs.as_ref().len();
    let width = num_lookup_cols::<WIDTH>();
    let height = output_count.next_power_of_two().max(1);
    let mut values = vec![<F as PrimeCharacteristicRing>::ZERO; width * height];

    let public_multiplicities = public_multiplicities(instance);
    let linear_multiplicities = linear_constraints.map(lin_multiplicities);
    let linear_lookup_multiplicities = linear_constraints.map(|constraints| {
        linear_lookup_multiplicities::<WIDTH>(
            input_outputs.as_ref(),
            &lin_multiplicities(constraints),
        )
    });
    let output_multiplicities = output_multiplicities(input_outputs.as_ref(), vars_count);
    let input_multiplicities = input_multiplicities(input_outputs.as_ref(), vars_count);
    let linear_inputs = linear_lookup_multiplicities
        .as_ref()
        .map(|(input, _)| input.clone())
        .unwrap_or_else(|| vec![[0; WIDTH]; output_count]);
    let linear_outputs = linear_lookup_multiplicities
        .as_ref()
        .map(|(_, output)| output.clone())
        .unwrap_or_else(|| vec![[0; WIDTH]; output_count]);

    for (row_idx, pair) in input_outputs.as_ref().iter().enumerate() {
        let output_mult = output_multiplicities[row_idx];
        let input_mult = input_multiplicities[row_idx];
        let linear_input = linear_inputs[row_idx];
        let linear_output = linear_outputs[row_idx];
        let offset = row_idx * width;
        let row = &mut values[offset..offset + width];
        let lookup: &mut LookupCols<F, WIDTH> = row.borrow_mut();
        lookup.input_public = pair
            .input
            .map(|var| F::from_bool(public_multiplicities[var.0].is_some()));
        lookup.output_public = pair
            .output
            .map(|var| F::from_bool(public_multiplicities[var.0].is_some()));
        lookup.input_vars = pair.input.map(|var| F::from_usize(var.0));
        lookup.output_vars = pair.output.map(|var| F::from_usize(var.0));
        lookup.output_multiplicities = output_mult.map(F::from_usize);
        lookup.input_multiplicities = input_mult.map(F::from_usize);
        lookup.input_linear_constraints = match &linear_multiplicities {
            Some(_) => linear_input.map(F::from_usize),
            None => core::array::from_fn(|_| <F as PrimeCharacteristicRing>::ZERO),
        };
        lookup.output_linear_constraints = match &linear_multiplicities {
            Some(_) => linear_output.map(F::from_usize),
            None => core::array::from_fn(|_| <F as PrimeCharacteristicRing>::ZERO),
        };
    }

    DenseMatrix::new(values, width)
}

fn build_public_lookup_table_trace<F, const WIDTH: usize>(
    instance: &PermutationInstanceBuilder<F, WIDTH>,
) -> DenseMatrix<F>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    let public_multiplicities = public_multiplicities(instance);
    let public_vars = instance.public_vars();
    let width = num_public_lookup_cols();
    let height = public_vars.len().next_power_of_two().max(1);
    let mut values = vec![<F as PrimeCharacteristicRing>::ZERO; width * height];

    for (row_idx, (var, val)) in public_vars.iter().enumerate() {
        let multiplicity = public_multiplicities[var.0].unwrap_or(0);
        let offset = row_idx * width;
        values[offset] = F::from_usize(var.0);
        values[offset + 1] = *val;
        values[offset + 2] = F::from_usize(multiplicity);
    }

    DenseMatrix::new(values, width)
}

fn build_public_lookup_main_trace<F>(trace_len: usize) -> DenseMatrix<F>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    assert!(trace_len.is_power_of_two());
    DenseMatrix::new(vec![<F as PrimeCharacteristicRing>::ZERO; trace_len], 1)
}

fn build_lin_lookup_trace<F, const WIDTH: usize, const LIN_WIDTH: usize>(
    instance: &PermutationInstanceBuilder<F, WIDTH>,
    lc: &LinearConstraintsInstance<F>,
) -> DenseMatrix<F>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    validate_linear_constraints::<LIN_WIDTH, _, _>(lc, "instance");
    let constraints_len = lc.as_ref().len();
    let width = num_linear_preprocessed_cols::<LIN_WIDTH>();
    let height = constraints_len.next_power_of_two().max(1);
    let mut values = vec![<F as PrimeCharacteristicRing>::ZERO; width * height];
    let _ = instance;

    for (row_idx, equation) in lc.as_ref().iter().enumerate() {
        let linear_vars = core::array::from_fn(|i| equation.linear_combination[i].1);
        let linear_multiplicities = core::array::from_fn(|i| {
            if equation.linear_combination[i].0 == <F as PrimeCharacteristicRing>::ZERO {
                <F as PrimeCharacteristicRing>::ZERO
            } else {
                <F as PrimeCharacteristicRing>::ONE
            }
        });
        let offset = row_idx * width;
        let row = &mut values[offset..offset + width];
        let column: &mut LinearConstraintPreprocessedCols<F, LIN_WIDTH> = row.borrow_mut();
        column.linear_vars = linear_vars.map(|var| F::from_usize(var.0));
        column.image_value = equation.image;
        column.linear_multiplicities = linear_multiplicities;
    }

    DenseMatrix::new(values, width)
}

fn build_linear_constraints_trace<F, const LIN_WIDTH: usize>(
    lc: &LinearConstraintsWitness<F>,
) -> DenseMatrix<F>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    validate_linear_constraints::<LIN_WIDTH, _, _>(lc, "witness");
    let constraints_len = lc.as_ref().len();
    let width = num_linear_main_cols::<LIN_WIDTH>();
    let height = constraints_len.next_power_of_two().max(1);
    let mut values = vec![<F as PrimeCharacteristicRing>::ZERO; width * height];

    for (row_idx, equation) in lc.as_ref().iter().enumerate() {
        let linear_coefficients = core::array::from_fn(|i| equation.linear_combination[i].0);
        let linear_values = core::array::from_fn(|i| equation.linear_combination[i].1);
        let offset = row_idx * width;
        let row = &mut values[offset..offset + width];
        let column: &mut LinearConstraintCols<F, LIN_WIDTH> = row.borrow_mut();
        column.linear_coefficients = linear_coefficients;
        column.linear_combination = linear_values;
        column.image = equation.image;
    }

    DenseMatrix::new(values, width)
}

fn compress_query_answer_pairs<F, const WIDTH: usize>(
    witness_rows: &[QueryAnswerPair<F, WIDTH>],
) -> Vec<CountedQueryAnswerPair<F, WIDTH>>
where
    F: Field + Unit + PartialEq + Send + Sync,
{
    let mut compressed: Vec<CountedQueryAnswerPair<F, WIDTH>> = Vec::new();

    for pair in witness_rows {
        if let Some(existing) = compressed.iter_mut().find(|row| row.pair == *pair) {
            existing.multiplicity += 1;
            continue;
        }
        compressed.push(CountedQueryAnswerPair {
            pair: pair.clone(),
            multiplicity: 1,
        });
    }

    compressed
}

fn value_lookup_elements<T, U, const WIDTH: usize>(
    input: &[T; WIDTH],
    output: &[T; WIDTH],
) -> Vec<U>
where
    T: Clone + Into<U>,
{
    input
        .iter()
        .cloned()
        .chain(output.iter().cloned())
        .map(Into::into)
        .collect()
}

fn output_multiplicities<const WIDTH: usize>(
    constraints: &[QueryAnswerPair<FieldVar, WIDTH>],
    vars_count: usize,
) -> Vec<[usize; WIDTH]> {
    let multiplicities = multiplicities(constraints, vars_count);
    constraints
        .iter()
        .map(|pair| pair.output.map(|var| multiplicities[var.0]))
        .collect()
}

fn input_multiplicities<const WIDTH: usize>(
    constraints: &[QueryAnswerPair<FieldVar, WIDTH>],
    vars_count: usize,
) -> Vec<[usize; WIDTH]> {
    let mut outputs = vec![false; vars_count];
    for pair in constraints {
        for var in pair.output {
            outputs[var.0] = true;
        }
    }

    constraints
        .iter()
        .map(|pair| pair.input.map(|var| usize::from(outputs[var.0])))
        .collect()
}

fn multiplicities<const WIDTH: usize>(
    constraints: &[QueryAnswerPair<FieldVar, WIDTH>],
    vars_count: usize,
) -> Vec<usize> {
    let mut mult = vec![0; vars_count];
    for input_var in constraints.iter().flat_map(|pair| pair.input.iter()) {
        mult[input_var.0] += 1;
    }
    mult
}

fn linear_lookup_multiplicities<const WIDTH: usize>(
    constraints: &[QueryAnswerPair<FieldVar, WIDTH>],
    linear_counts: &[Option<usize>],
) -> (Vec<[usize; WIDTH]>, Vec<[usize; WIDTH]>) {
    let mut remaining = linear_counts
        .iter()
        .map(|count| count.unwrap_or(0))
        .collect::<Vec<_>>();
    let mut input_multiplicities = Vec::with_capacity(constraints.len());
    let mut output_multiplicities = Vec::with_capacity(constraints.len());

    for pair in constraints {
        let mut input_counts = [0; WIDTH];
        let mut output_counts = [0; WIDTH];

        for (slot, var) in input_counts.iter_mut().zip(pair.input.iter()) {
            let count = remaining.get_mut(var.0).map_or(0, core::mem::take);
            *slot = count;
        }
        for (slot, var) in output_counts.iter_mut().zip(pair.output.iter()) {
            let count = remaining.get_mut(var.0).map_or(0, core::mem::take);
            *slot = count;
        }

        input_multiplicities.push(input_counts);
        output_multiplicities.push(output_counts);
    }

    debug_assert!(remaining.into_iter().all(|count| count == 0));
    (input_multiplicities, output_multiplicities)
}

fn public_multiplicities<F, const WIDTH: usize>(
    instance: &PermutationInstanceBuilder<F, WIDTH>,
) -> Vec<Option<usize>>
where
    F: Field + Unit + PartialEq,
{
    let public = instance.allocator().public_vars();
    let mut mult = vec![None; instance.allocator().vars_count()];

    for (var, _) in public.iter() {
        mult[var.0] = Some(0);
    }

    for var in instance
        .constraints()
        .as_ref()
        .iter()
        .flat_map(|pair| pair.input.iter().chain(pair.output.iter()))
    {
        mult[var.0] = mult[var.0].map(|count| count + 1);
    }

    mult
}

fn lin_multiplicities<F>(lc: &LinearConstraintsInstance<F>) -> Vec<Option<usize>>
where
    F: Field + Unit + PartialEq,
{
    let vars_count = lc
        .as_ref()
        .iter()
        .flat_map(|equation| equation.linear_combination.iter().map(|(_, var)| var.0))
        .max()
        .map(|max_var| max_var + 1)
        .unwrap_or(0);
    let mut mult = vec![None; vars_count];

    for equation in lc.as_ref() {
        for (coeff, var) in &equation.linear_combination {
            if *coeff != <F as PrimeCharacteristicRing>::ZERO {
                mult[var.0] = Some(mult[var.0].unwrap_or(0) + 1);
            }
        }
    }

    mult
}

fn validate_linear_constraints<const LIN_WIDTH: usize, T, U>(
    lc: &LinearConstraints<T, U>,
    source: &str,
) {
    for (idx, equation) in lc.as_ref().iter().enumerate() {
        assert_eq!(
            equation.linear_combination.len(),
            LIN_WIDTH,
            "{source} linear equation {idx} must have exactly {LIN_WIDTH} terms",
        );
    }
}

fn pad_dense_matrix_to_height<T: Clone + Default + Send + Sync>(
    mut matrix: DenseMatrix<T>,
    target_height: usize,
) -> DenseMatrix<T> {
    let width = matrix.width;
    let current_height = matrix.values.len() / width;
    if current_height < target_height {
        matrix.values.resize_with(target_height * width, T::default);
    }
    DenseMatrix::new(matrix.values, width)
}

fn trace_degree_bits<T: Clone + Send + Sync>(traces: &[DenseMatrix<T>]) -> Vec<usize> {
    traces
        .iter()
        .map(|trace| {
            let trace_height = trace.height();
            assert!(trace_height.is_power_of_two());
            trace_height.trailing_zeros() as usize
        })
        .collect()
}

fn log_ext_degrees<SC: StarkGenericConfig>(log_degrees: &[usize], config: &SC) -> Vec<usize> {
    log_degrees
        .iter()
        .map(|&degree| degree + config.is_zk())
        .collect()
}

#[cfg(test)]
mod multiplicity_tests {
    use super::generate_trace_rows_with_linear;
    use crate::{
        poseidon2::{BabyBearPoseidon2Backend, BabyBearPoseidon2_16HashAir, POSEIDON2_16_WIDTH},
        relation, RelationField,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use spongefish_circuit::permutation::{PermutationInstanceBuilder, PermutationWitnessBuilder};
    use spongefish_poseidon2::BabyBearPoseidon2_16;

    #[test]
    fn repeated_witness_rows_share_one_hash_trace() {
        type B = BabyBearPoseidon2Backend;

        let hash = BabyBearPoseidon2_16HashAir::default();
        let permutation = BabyBearPoseidon2_16::default();
        let instance = PermutationInstanceBuilder::<RelationField<B>, POSEIDON2_16_WIDTH>::new();
        let witness = PermutationWitnessBuilder::<_, POSEIDON2_16_WIDTH>::new(permutation);
        let input = core::array::from_fn(|idx| RelationField::<B>::from_usize(idx + 1));

        for _ in 0..2 {
            let input_vars = instance.allocator().allocate_vars::<POSEIDON2_16_WIDTH>();
            let _ = instance.allocate_permutation(&input_vars);
            let _ = witness.allocate_permutation(&input);
        }

        let (_airs, traces) =
            generate_trace_rows_with_linear::<_, RelationField<B>, _, POSEIDON2_16_WIDTH, 1>(
                &hash, &instance, &witness,
            );

        assert_eq!(traces[0].height(), 1);
        assert_eq!(traces[2].height(), 2);

        let backend = B::default();
        let proof =
            relation::prove::<B, _, _, POSEIDON2_16_WIDTH, 1>(&backend, &hash, &instance, &witness);
        assert!(relation::verify::<B, _, POSEIDON2_16_WIDTH, 1>(
            &backend, &hash, &instance, &proof
        )
        .is_ok());
    }
}
