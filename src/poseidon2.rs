use alloc::vec::Vec;
use core::{borrow::Borrow, marker::PhantomData};

use crate::{HashInvocationAir, QueryAnswerPair, StarkRelationBackend};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{
    default_babybear_poseidon2_16, BabyBear, GenericPoseidon2LinearLayersBabyBear,
    Poseidon2BabyBear, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16, BABYBEAR_S_BOX_DEGREE,
};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::{extension::BinomialExtensionField, Field, PrimeField};
use p3_fri::{create_benchmark_fri_params_zk, HidingFriPcs};
use p3_koala_bear::{
    default_koalabear_poseidon2_16, GenericPoseidon2LinearLayersKoalaBear, KoalaBear,
    Poseidon2KoalaBear, KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16, KOALABEAR_S_BOX_DEGREE,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeHidingMmcs;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{Poseidon2Air, Poseidon2Cols, RoundConstants};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, StarkGenericConfig};
use rand::{rngs::SmallRng, SeedableRng};

pub const POSEIDON2_16_WIDTH: usize = 16;
pub const POSEIDON2_16_HALF_FULL_ROUNDS: usize = 4;

pub const BABYBEAR_POSEIDON2_SBOX_REGISTERS: usize = 1;

pub const KOALABEAR_POSEIDON2_SBOX_REGISTERS: usize = 0;

#[derive(Clone)]
pub struct BabyBearPoseidon2_16(Poseidon2BabyBear<POSEIDON2_16_WIDTH>);

impl From<Poseidon2BabyBear<POSEIDON2_16_WIDTH>> for BabyBearPoseidon2_16 {
    fn from(inner: Poseidon2BabyBear<POSEIDON2_16_WIDTH>) -> Self {
        Self(inner)
    }
}

impl Default for BabyBearPoseidon2_16 {
    fn default() -> Self {
        Self(default_babybear_poseidon2_16())
    }
}

impl spongefish::Permutation<POSEIDON2_16_WIDTH> for BabyBearPoseidon2_16
where
    Poseidon2BabyBear<POSEIDON2_16_WIDTH>:
        p3_symmetric::Permutation<[BabyBear; POSEIDON2_16_WIDTH]>,
{
    type U = BabyBear;

    fn permute(&self, state: &[Self::U; POSEIDON2_16_WIDTH]) -> [Self::U; POSEIDON2_16_WIDTH] {
        p3_symmetric::Permutation::permute(&self.0, *state)
    }

    fn permute_mut(&self, state: &mut [Self::U; POSEIDON2_16_WIDTH]) {
        p3_symmetric::Permutation::permute_mut(&self.0, state);
    }
}

#[derive(Clone)]
pub struct KoalaBearPoseidon2_16(Poseidon2KoalaBear<POSEIDON2_16_WIDTH>);

impl From<Poseidon2KoalaBear<POSEIDON2_16_WIDTH>> for KoalaBearPoseidon2_16 {
    fn from(inner: Poseidon2KoalaBear<POSEIDON2_16_WIDTH>) -> Self {
        Self(inner)
    }
}

impl Default for KoalaBearPoseidon2_16 {
    fn default() -> Self {
        Self(default_koalabear_poseidon2_16())
    }
}

impl spongefish::Permutation<POSEIDON2_16_WIDTH> for KoalaBearPoseidon2_16
where
    Poseidon2KoalaBear<POSEIDON2_16_WIDTH>:
        p3_symmetric::Permutation<[KoalaBear; POSEIDON2_16_WIDTH]>,
{
    type U = KoalaBear;

    fn permute(&self, state: &[Self::U; POSEIDON2_16_WIDTH]) -> [Self::U; POSEIDON2_16_WIDTH] {
        p3_symmetric::Permutation::permute(&self.0, *state)
    }

    fn permute_mut(&self, state: &mut [Self::U; POSEIDON2_16_WIDTH]) {
        p3_symmetric::Permutation::permute_mut(&self.0, state);
    }
}

type Poseidon2_16Air<
    F,
    LinearLayers,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
> = Poseidon2Air<
    F,
    LinearLayers,
    POSEIDON2_16_WIDTH,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    POSEIDON2_16_HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS,
>;

type Poseidon2_16Cols<
    T,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
> = Poseidon2Cols<
    T,
    POSEIDON2_16_WIDTH,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    POSEIDON2_16_HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS,
>;

type Poseidon2_16RoundConstants<F, const PARTIAL_ROUNDS: usize> =
    RoundConstants<F, POSEIDON2_16_WIDTH, POSEIDON2_16_HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

#[doc(hidden)]
pub trait Poseidon2FieldConfig<
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
>: Clone + Copy + Default + 'static
{
    type F: Field + PrimeField;
    type LinearLayers: GenericPoseidon2LinearLayers<POSEIDON2_16_WIDTH> + Sync;
    type Config: StarkGenericConfig;

    fn config(seed: u64) -> Self::Config;

    fn round_constants() -> Poseidon2_16RoundConstants<Self::F, PARTIAL_ROUNDS>;
}

#[derive(Clone, Copy, Default)]
pub struct BabyBearPoseidon2Config;

#[derive(Clone, Copy, Default)]
pub struct KoalaBearPoseidon2Config;

#[derive(Clone, Copy, Default)]
pub struct Poseidon2Backend<
    C,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
>(PhantomData<C>)
where
    C: Poseidon2FieldConfig<SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>;

pub struct Poseidon2_16HashAir<
    C,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
> where
    C: Poseidon2FieldConfig<SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
{
    air: Poseidon2_16Air<C::F, C::LinearLayers, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
    _marker: PhantomData<C>,
}

impl<C, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize, const PARTIAL_ROUNDS: usize> Clone
    for Poseidon2_16HashAir<C, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
where
    C: Poseidon2FieldConfig<SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
{
    fn clone(&self) -> Self {
        Self {
            air: self.air.clone(),
            _marker: PhantomData,
        }
    }
}

pub type BabyBearPoseidon2_16Air = Poseidon2_16Air<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    BABYBEAR_S_BOX_DEGREE,
    BABYBEAR_POSEIDON2_SBOX_REGISTERS,
    BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
>;

pub type KoalaBearPoseidon2_16Air = Poseidon2_16Air<
    KoalaBear,
    GenericPoseidon2LinearLayersKoalaBear,
    KOALABEAR_S_BOX_DEGREE,
    KOALABEAR_POSEIDON2_SBOX_REGISTERS,
    KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16,
>;

pub type BabyBearPoseidon2_16RoundConstants =
    Poseidon2_16RoundConstants<BabyBear, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16>;
pub type KoalaBearPoseidon2_16RoundConstants =
    Poseidon2_16RoundConstants<KoalaBear, KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16>;

pub type BabyBearPoseidon2Challenge = BinomialExtensionField<BabyBear, 4>;
pub type BabyBearTranscriptPerm = Poseidon2BabyBear<POSEIDON2_16_WIDTH>;
pub type BabyBearTranscriptHash =
    PaddingFreeSponge<BabyBearTranscriptPerm, POSEIDON2_16_WIDTH, 8, 8>;
pub type BabyBearTranscriptCompress =
    TruncatedPermutation<BabyBearTranscriptPerm, 2, 8, POSEIDON2_16_WIDTH>;
pub type BabyBearValMmcs = MerkleTreeHidingMmcs<
    <BabyBear as Field>::Packing,
    <BabyBear as Field>::Packing,
    BabyBearTranscriptHash,
    BabyBearTranscriptCompress,
    SmallRng,
    2,
    8,
    4,
>;
pub type BabyBearChallengeMmcs =
    ExtensionMmcs<BabyBear, BabyBearPoseidon2Challenge, BabyBearValMmcs>;
pub type BabyBearChallenger =
    DuplexChallenger<BabyBear, BabyBearTranscriptPerm, POSEIDON2_16_WIDTH, 8>;
pub type BabyBearDft = p3_dft::Radix2DitParallel<BabyBear>;
pub type BabyBearPcs =
    HidingFriPcs<BabyBear, BabyBearDft, BabyBearValMmcs, BabyBearChallengeMmcs, SmallRng>;
pub type BabyBearPoseidon2StarkConfig =
    StarkConfig<BabyBearPcs, BabyBearPoseidon2Challenge, BabyBearChallenger>;

pub type KoalaBearPoseidon2Challenge = BinomialExtensionField<KoalaBear, 4>;
pub type KoalaBearTranscriptPerm = Poseidon2KoalaBear<POSEIDON2_16_WIDTH>;
pub type KoalaBearTranscriptHash =
    PaddingFreeSponge<KoalaBearTranscriptPerm, POSEIDON2_16_WIDTH, 8, 8>;
pub type KoalaBearTranscriptCompress =
    TruncatedPermutation<KoalaBearTranscriptPerm, 2, 8, POSEIDON2_16_WIDTH>;
pub type KoalaBearValMmcs = MerkleTreeHidingMmcs<
    <KoalaBear as Field>::Packing,
    <KoalaBear as Field>::Packing,
    KoalaBearTranscriptHash,
    KoalaBearTranscriptCompress,
    SmallRng,
    2,
    8,
    4,
>;
pub type KoalaBearChallengeMmcs =
    ExtensionMmcs<KoalaBear, KoalaBearPoseidon2Challenge, KoalaBearValMmcs>;
pub type KoalaBearChallenger =
    DuplexChallenger<KoalaBear, KoalaBearTranscriptPerm, POSEIDON2_16_WIDTH, 8>;
pub type KoalaBearDft = p3_dft::Radix2DitParallel<KoalaBear>;
pub type KoalaBearPcs =
    HidingFriPcs<KoalaBear, KoalaBearDft, KoalaBearValMmcs, KoalaBearChallengeMmcs, SmallRng>;
pub type KoalaBearPoseidon2StarkConfig =
    StarkConfig<KoalaBearPcs, KoalaBearPoseidon2Challenge, KoalaBearChallenger>;

pub type BabyBearPoseidon2Backend = Poseidon2Backend<
    BabyBearPoseidon2Config,
    BABYBEAR_S_BOX_DEGREE,
    BABYBEAR_POSEIDON2_SBOX_REGISTERS,
    BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
>;

pub type KoalaBearPoseidon2Backend = Poseidon2Backend<
    KoalaBearPoseidon2Config,
    KOALABEAR_S_BOX_DEGREE,
    KOALABEAR_POSEIDON2_SBOX_REGISTERS,
    KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16,
>;

pub type BabyBearPoseidon2_16HashAir = Poseidon2_16HashAir<
    BabyBearPoseidon2Config,
    BABYBEAR_S_BOX_DEGREE,
    BABYBEAR_POSEIDON2_SBOX_REGISTERS,
    BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
>;

pub type KoalaBearPoseidon2_16HashAir = Poseidon2_16HashAir<
    KoalaBearPoseidon2Config,
    KOALABEAR_S_BOX_DEGREE,
    KOALABEAR_POSEIDON2_SBOX_REGISTERS,
    KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16,
>;

impl
    Poseidon2FieldConfig<
        BABYBEAR_S_BOX_DEGREE,
        BABYBEAR_POSEIDON2_SBOX_REGISTERS,
        BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
    > for BabyBearPoseidon2Config
{
    type F = BabyBear;
    type LinearLayers = GenericPoseidon2LinearLayersBabyBear;
    type Config = BabyBearPoseidon2StarkConfig;

    fn config(seed: u64) -> Self::Config {
        let perm = default_babybear_poseidon2_16();
        let hash = BabyBearTranscriptHash::new(perm.clone());
        let compress = BabyBearTranscriptCompress::new(perm.clone());
        let val_mmcs = BabyBearValMmcs::new(hash, compress, 0, SmallRng::seed_from_u64(seed));
        let challenge_mmcs = BabyBearChallengeMmcs::new(val_mmcs.clone());
        let fri_params = create_benchmark_fri_params_zk(challenge_mmcs);
        let pcs = BabyBearPcs::new(
            BabyBearDft::default(),
            val_mmcs,
            fri_params,
            4,
            SmallRng::seed_from_u64(1),
        );
        let challenger = BabyBearChallenger::new(perm);
        StarkConfig::new(pcs, challenger)
    }

    fn round_constants() -> BabyBearPoseidon2_16RoundConstants {
        RoundConstants::new(
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_INTERNAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        )
    }
}

impl
    Poseidon2FieldConfig<
        KOALABEAR_S_BOX_DEGREE,
        KOALABEAR_POSEIDON2_SBOX_REGISTERS,
        KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16,
    > for KoalaBearPoseidon2Config
{
    type F = KoalaBear;
    type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
    type Config = KoalaBearPoseidon2StarkConfig;

    fn config(seed: u64) -> Self::Config {
        let perm = default_koalabear_poseidon2_16();
        let hash = KoalaBearTranscriptHash::new(perm.clone());
        let compress = KoalaBearTranscriptCompress::new(perm.clone());
        let val_mmcs = KoalaBearValMmcs::new(hash, compress, 0, SmallRng::seed_from_u64(seed));
        let challenge_mmcs = KoalaBearChallengeMmcs::new(val_mmcs.clone());
        let fri_params = create_benchmark_fri_params_zk(challenge_mmcs);
        let pcs = KoalaBearPcs::new(
            KoalaBearDft::default(),
            val_mmcs,
            fri_params,
            4,
            SmallRng::seed_from_u64(1),
        );
        let challenger = KoalaBearChallenger::new(perm);
        StarkConfig::new(pcs, challenger)
    }

    fn round_constants() -> KoalaBearPoseidon2_16RoundConstants {
        RoundConstants::new(
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_INTERNAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        )
    }
}

impl<C, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize, const PARTIAL_ROUNDS: usize> Default
    for Poseidon2_16HashAir<C, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
where
    C: Poseidon2FieldConfig<SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize, const PARTIAL_ROUNDS: usize>
    Poseidon2_16HashAir<C, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
where
    C: Poseidon2FieldConfig<SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
{
    #[must_use]
    pub fn new() -> Self {
        Self {
            air: Poseidon2_16Air::new(C::round_constants()),
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn air(
        &self,
    ) -> &Poseidon2_16Air<C::F, C::LinearLayers, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS> {
        &self.air
    }
}

impl<C, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize, const PARTIAL_ROUNDS: usize>
    StarkRelationBackend for Poseidon2Backend<C, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
where
    C: Poseidon2FieldConfig<SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
{
    type Config = C::Config;

    fn config(&self, seed: u64) -> Self::Config {
        C::config(seed)
    }
}

impl<C, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize, const PARTIAL_ROUNDS: usize>
    HashInvocationAir<C::F, POSEIDON2_16_WIDTH>
    for Poseidon2_16HashAir<C, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
where
    C: Poseidon2FieldConfig<SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
{
    type Frame<'a, Var>
        = &'a [Var]
    where
        Self: 'a,
        Var: 'a;

    fn main_width(&self) -> usize {
        BaseAir::<C::F>::width(&self.air)
    }

    fn eval<AB>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = C::F>,
    {
        Air::<AB>::eval(&self.air, builder);
    }

    fn row_frame<'a, Var>(&self, row: &'a [Var]) -> Self::Frame<'a, Var> {
        row
    }

    fn build_trace(
        &self,
        witness_rows: &[QueryAnswerPair<C::F, POSEIDON2_16_WIDTH>],
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<C::F> {
        let inputs = witness_rows
            .iter()
            .map(|pair| pair.input)
            .collect::<Vec<_>>();

        p3_poseidon2_air::generate_trace_rows::<
            C::F,
            C::LinearLayers,
            POSEIDON2_16_WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            POSEIDON2_16_HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs, &C::round_constants(), extra_capacity_bits)
    }

    fn invocation<AB>(
        &self,
        frame: &Self::Frame<'_, AB::Var>,
    ) -> QueryAnswerPair<AB::Expr, POSEIDON2_16_WIDTH>
    where
        AB: AirBuilder<F = C::F>,
    {
        let cols: &Poseidon2_16Cols<_, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS> =
            (*frame).borrow();

        QueryAnswerPair::new(
            cols.inputs.map(Into::into),
            cols.ending_full_rounds[POSEIDON2_16_HALF_FULL_ROUNDS - 1]
                .post
                .map(Into::into),
        )
    }
}
