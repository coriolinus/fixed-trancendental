use std::ops::{BitAnd, ShrAssign};

use fixed::traits::{Fixed, ToFixed};

/// A selection of transcendental functions.
pub trait Transcendental {
    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> Self;

    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> Self;

    /// Returns the natural logarithm of the number.
    fn ln(self) -> Self;

    /// Raises the number to a non-integral power.
    fn pow(self, n: Self) -> Self;

    /// Raises the number to an integer power.
    ///
    /// This may be faster than [`self.pow`].
    fn powi(self, n: u64) -> Self;

    /// Returns `e^(self)`, the exponential function.
    fn exp(self) -> Self;

    /// Returns `2^(self)`.
    fn exp2(self) -> Self;
}

/// Returns the base 2 logarithm of the number.
///
/// Implementation of the Fast Binary Logarithm Algorithm by Clay S. Turner,
/// published in IEEE Signal Processing Magazine, September 2010, page 140.
///
/// http://www.claysturner.com/dsp/BinaryLogarithm.pdf
pub fn log2<N: Fixed>(mut x: N) -> N {
    let mut y = N::ZERO;
    let mut b: N = 0.5_f32.wrapping_to_fixed();

    let one = 1.saturating_to_fixed::<N>();
    let two = 2.saturating_to_fixed::<N>();

    while x < one {
        x <<= 1;
        y -= one;
    }
    while x >= two {
        x >>= 1;
        y += one;
    }

    debug_assert!(one <= x);
    debug_assert!(x < two);

    for _ in 0..N::FRAC_NBITS {
        x *= x;
        if x >= two {
            x >>= 1;
            y += b;
        }
        b >>= 1;
    }

    y
}

pub trait MaybeNeg {
    /// Negate `self` if the underlying type is signed.
    fn neg(&mut self);
}

macro_rules! impl_maybe_neg {
    (
        signed: $($signed:ty),*;
        unsigned: $($unsigned:ty),*;
    ) => {
        $(
            impl MaybeNeg for $signed {
                fn neg(&mut self) {
                    *self = -*self;
                }
            }
        )*
        $(
            impl MaybeNeg for $unsigned {
                fn neg(&mut self) {
                    // nothing
                }
            }
        )*
    };
}

impl_maybe_neg!(
    signed: isize, i8, i16, i32, i64, i128, f32, f64;
    unsigned: usize, u8, u16, u32, u64, u128;
);

/// Returns `self^(n)`.
///
/// Uses exponentiation by squaring.
pub fn powi<N>(mut x: N, mut pow: N::Bits) -> N
where
    N: Fixed,
    N::Bits: From<u8> + Copy + Ord + MaybeNeg + BitAnd<Output = N::Bits> + ShrAssign<u8>,
{
    let zero = 0.into();
    let one = 1.into();

    if pow < zero {
        x = x.recip();
        pow.neg();
    }
    if pow == zero {
        return 1.saturating_to_fixed();
    }
    let mut y: N = 1.saturating_to_fixed();
    while pow > one {
        if pow & one != zero {
            y *= x;
        }
        x *= x;
        pow >>= 1;
    }
    x * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixed::{
        traits::FromFixed,
        types::{I16F16, I32F32, I64F64, I8F8, U16F16, U32F32, U64F64, U8F8},
    };
    use rstest::rstest;

    macro_rules! test_mods {
        ($($fixed:ident),*) => {
            $(
                paste::paste!{
                    mod [<$fixed:snake>] {
                        use super::*;

                        #[rstest]
                        #[case(f32::MAX)]
                        #[case(1.0)]
                        #[case(10.0)]
                        #[case(100.0)]
                        #[case(255.0)]
                        #[case(256.0)]
                        #[case(257.0)]
                        #[case(4095.0)]
                        #[case(4096.0)]
                        #[case(4097.0)]
                        fn test_log2_f32(#[case] n: f32) {
                            if let Some(fixed) = n.checked_to_fixed::<$fixed>() {
                                let delta: f32 = f32::from_fixed($fixed::DELTA);
                                dbg!(n, delta);
                                let float_log2 = n.log2();
                                let fixed_log2 = f32::from_fixed(dbg!(log2(fixed)));
                                dbg!(float_log2, fixed_log2);
                                let dist = dbg!((fixed_log2 - float_log2).abs());
                                assert!(dist <= delta);
                            }
                        }

                        #[rstest]
                        #[case(f64::MAX)]
                        #[case(1.0)]
                        #[case(10.0)]
                        #[case(100.0)]
                        #[case(255.0)]
                        #[case(256.0)]
                        #[case(257.0)]
                        #[case(4095.0)]
                        #[case(4096.0)]
                        #[case(4097.0)]
                        fn test_log2_f64(#[case] n: f64) {
                            if let Some(fixed) = n.checked_to_fixed::<$fixed>() {
                                let delta = f64::from_fixed($fixed::DELTA);
                                dbg!(n, delta);
                                let float_log2 = n.log2();
                                let fixed_log2 = f64::from_fixed(dbg!(log2(fixed)));
                                dbg!(float_log2, fixed_log2);
                                let dist = dbg!((fixed_log2 - float_log2).abs());
                                assert!(dist <= delta);
                            }
                        }

                        #[rstest]
                        #[case(u64::MAX)]
                        #[case(1)]
                        #[case(10)]
                        #[case(100)]
                        #[case(255)]
                        #[case(256)]
                        #[case(257)]
                        #[case(4095)]
                        #[case(4096)]
                        #[case(4097)]
                        fn test_log2_u64(#[case] n: u64) {
                            if let Some(fixed) = n.checked_to_fixed::<$fixed>() {
                                let float_log2 = (n as f64).log2();
                                let log2_value = float_log2.floor() as u64;
                                let fixed_log2 = u64::from_fixed(dbg!(log2(fixed)));
                                dbg!(log2_value, fixed_log2);
                            }
                        }

                        // Note that as of version `1.9.0` of `fixed`, this test fails because
                        // multiplication does not reliably produce accurate results.
                        //
                        // https://gitlab.com/tspiteri/fixed/-/issues/33
                        #[rstest]
                        fn test_powi(
                            #[values(
                                1.0/12345.6,
                                1.0/1234.5,
                                1.0/123.4,
                                1.0/12.3,
                                1.0/1.2,
                                1.0,
                                1.2,
                                12.3,
                                123.4,
                                1234.5,
                                12345.6,
                            )]
                            base: f64,
                            #[values(
                                1,
                                2,
                                3,
                                4,
                                5,
                                10,
                                16,
                                32,
                            )]
                            power: u8,
                        ) {
                            if let Some(fixed) = base.checked_to_fixed::<$fixed>() {
                                dbg!(base, power, fixed);
                                let expect = dbg!(base.powi(power.into()));
                                if expect >= f64::from_fixed($fixed::MAX) {
                                    return; // will overflow
                                }
                                let computed = dbg!(powi(fixed, power.into()));
                                dbg!($fixed::from_bits(computed.to_bits()-1));
                                let diff = (expect - f64::from_fixed(computed)).abs();
                                assert!(dbg!(diff) <= dbg!(f64::from_fixed(dbg!($fixed::DELTA))));
                            }
                        }
                    }
                }
            )*
        };
    }

    test_mods!(I8F8, I16F16, I32F32, I64F64, U8F8, U16F16, U32F32, U64F64);
}
