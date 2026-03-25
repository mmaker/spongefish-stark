use alloc::{vec, vec::Vec};
use core::{borrow::Borrow, borrow::BorrowMut, mem::size_of};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::LookupAir;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use spongefish::Unit;

use crate::{HashInvocationAir, QueryAnswerPair};

#[repr(C)]
struct OutputPreprocessedCols<T, const WIDTH: usize> {
    selectors: [T; WIDTH],
    expected: [T; WIDTH],
}

impl<T, const WIDTH: usize> Borrow<OutputPreprocessedCols<T, WIDTH>> for [T] {
    fn borrow(&self) -> &OutputPreprocessedCols<T, WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to::<OutputPreprocessedCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize> BorrowMut<OutputPreprocessedCols<T, WIDTH>> for [T] {
    fn borrow_mut(&mut self) -> &mut OutputPreprocessedCols<T, WIDTH> {
        let (prefix, shorts, suffix) =
            unsafe { self.align_to_mut::<OutputPreprocessedCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

const fn num_output_preprocessed_cols<const WIDTH: usize>() -> usize {
    size_of::<OutputPreprocessedCols<u8, WIDTH>>()
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

/// Proves knowledge of hash preimages while only exposing selected output lanes.
#[derive(Clone)]
pub struct HashPreimageAir<H, F, const WIDTH: usize> {
    hash: H,
    public_outputs: Vec<[Option<F>; WIDTH]>,
}

impl<H, F, const WIDTH: usize> HashPreimageAir<H, F, WIDTH> {
    #[must_use]
    pub fn new(hash: H, public_outputs: Vec<[Option<F>; WIDTH]>) -> Self {
        Self {
            hash,
            public_outputs,
        }
    }

    #[must_use]
    pub fn hash(&self) -> &H {
        &self.hash
    }

    #[must_use]
    pub fn public_outputs(&self) -> &[[Option<F>; WIDTH]] {
        &self.public_outputs
    }
}

impl<H, F, const WIDTH: usize> HashPreimageAir<H, F, WIDTH>
where
    H: HashInvocationAir<F, WIDTH>,
    F: Field + Unit,
{
    #[must_use]
    pub fn build_trace_rows(
        &self,
        witness_rows: &[QueryAnswerPair<F, WIDTH>],
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        self.hash.build_trace(witness_rows, extra_capacity_bits)
    }
}

impl<H, F, const WIDTH: usize> BaseAir<F> for HashPreimageAir<H, F, WIDTH>
where
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn width(&self) -> usize {
        self.hash.main_width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows_per_invocation = self.hash.trace_rows_per_invocation().max(1);
        let logical_rows = self.public_outputs.len().max(1);
        let height = (logical_rows * rows_per_invocation).next_power_of_two();
        let width = num_output_preprocessed_cols::<WIDTH>();
        let mut values = vec![<F as PrimeCharacteristicRing>::ZERO; width * height];

        for (invocation_idx, outputs) in self.public_outputs.iter().enumerate() {
            let base_row = invocation_idx * rows_per_invocation;
            let expected =
                outputs.map(|value| value.unwrap_or(<F as PrimeCharacteristicRing>::ZERO));
            for row_offset in 0..rows_per_invocation {
                let row_idx = base_row + row_offset;
                let row = &mut values[row_idx * width..(row_idx + 1) * width];
                let cols: &mut OutputPreprocessedCols<F, WIDTH> = row.borrow_mut();
                cols.selectors = core::array::from_fn(|lane| {
                    if row_offset + 1 == rows_per_invocation && outputs[lane].is_some() {
                        <F as PrimeCharacteristicRing>::ONE
                    } else {
                        <F as PrimeCharacteristicRing>::ZERO
                    }
                });
                cols.expected = expected;
            }
        }

        Some(pad_dense_matrix_to_height(
            DenseMatrix::new(values, width),
            height,
        ))
    }
}

impl<AB, H, F, const WIDTH: usize> Air<AB> for HashPreimageAir<H, F, WIDTH>
where
    AB: AirBuilder<F = F>,
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
    fn eval(&self, builder: &mut AB) {
        self.hash.eval(builder);

        let main = builder.main();
        let local = main.current_slice();
        let frame = self.hash.row_frame(local);
        let invocation = self.hash.invocation::<AB>(&frame);

        let preprocessed = builder.preprocessed();
        let preprocessed_row = preprocessed.current_slice();
        let lane_values = {
            let columns: &OutputPreprocessedCols<_, WIDTH> = preprocessed_row.borrow();
            (0..WIDTH)
                .map(|lane| {
                    (
                        columns.selectors[lane].into(),
                        columns.expected[lane].into(),
                    )
                })
                .collect::<Vec<_>>()
        };

        for (lane, (selector, expected)) in lane_values.into_iter().enumerate() {
            builder.assert_eq(
                selector.clone() * invocation.output[lane].clone(),
                selector * expected,
            );
        }
    }
}

impl<H, F, const WIDTH: usize> LookupAir<F> for HashPreimageAir<H, F, WIDTH>
where
    H: HashInvocationAir<F, WIDTH> + Sync,
    F: Field + Unit + PartialEq + Send + Sync,
{
}
