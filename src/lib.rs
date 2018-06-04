extern crate rand;

use std::error::Error;
use std::fmt;

// Utility

#[derive(Debug, Clone)]
struct VectorDimMismatchError;
impl Error for VectorDimMismatchError {
    fn description(&self) -> &str {
        "Attempted to perform an operation using two vectors with different orders."
    }
}
impl fmt::Display for VectorDimMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Attempted to perform an operation using two vectors with different orders."
        )
    }
}

/// Adds two vectors and returns their sum
fn vector_add(a: &Vec<f64>, b: &Vec<f64>) -> Result<Vec<f64>, VectorDimMismatchError> {
    if a.len() != b.len() {
        return Err(VectorDimMismatchError);
    }
    let mut result: Vec<f64> = Vec::new();
    for (i, &n) in a.iter().enumerate() {
        result.push(n + b[i]);
    }
    Ok(result)
}

/// Multiplies each element in a vector by a constant and returns the product vector
fn vector_multiply(v: &Vec<f64>, c: &f64) -> Vec<f64> {
    v.iter().map(|x| x * c).collect()
}

/// Returns the euclidean distance between two Vec<64>s
fn dist64(a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, VectorDimMismatchError> {
    if a.len() != b.len() {
        return Err(VectorDimMismatchError);
    }
    let mut sum = 0.0;
    for (i, &n) in a.iter().enumerate() {
        sum += (n - b[i]).powf(2.0);
    }
    Ok(sum.powf(0.5))
}

// Mapping

/// A struct for defining a mapping of a point in space A to a point in space B. Spaces A and B may or may not have the same number of dimensions. The mapping has a weight that is used for conversion calculations

#[derive(Debug, Clone, PartialEq)]
pub struct Mapping {
    input: Vec<f64>,
    output: Vec<f64>,
    weight: f64,
}

impl Mapping {
    /// Initializes a mapping's input and output points
    pub fn from_points(input: Vec<f64>, output: Vec<f64>) -> Mapping {
        let new_mapping = Mapping {
            input: input,
            output: output,
            weight: 0.5,
        };
        new_mapping
    }
}

// Conversion

/// A struct that defines a conversion between two spaces. It is essentially a collection of mappings, all of whom's inputs and outputs fall within the bounds of the conversion's spaces. The conversion can evaluate arbitrary inputs for its output based on the conversion's mappings and its intermap function.

#[derive(Debug, Clone, PartialEq)]
pub struct Conversion {
    input_space: Vec<(f64, f64)>,
    output_space: Vec<(f64, f64)>,
    mappings: Vec<Mapping>,
    power: f64,
}

impl Conversion {
    /// Creates a new conversion from input and output spaces
    pub fn from_io_spaces(
        input_space: Vec<(f64, f64)>,
        output_space: Vec<(f64, f64)>,
    ) -> Conversion {
        let conv = Conversion {
            input_space,
            output_space,
            mappings: Vec::new(),
            power: 2.0,
        };
        conv
    }
    /// Initializes the conversion's power factor
    pub fn with_power(mut self, power: f64) -> Conversion {
        self.power = power;
        self
    }
    /// Adds a user-defined mapping to the conevrsion
    pub fn add_mapping(&mut self, m: Mapping) {
        if m.input.len() != self.input_space.len() {
            panic!("The mapping's input size is not equal to the conversion's input size!");
        }
        if m.output.len() != self.output_space.len() {
            panic!("The mapping's output size is not equal to the conversion's output size!");
        }

        self.mappings.push(m);
    }
    /// Evaluates an input's vector and returns the optional output based on the conversion's mappings and its intermap function
    pub fn convert(&self, in_value: Vec<f64>) -> Vec<f64> {
        // Find the maximum distance
        let mut max_dist = 0.0;
        for i in self.mappings.iter() {
            let dist = dist64(&in_value, &i.input).unwrap();
            if dist == 0.0 {
                return i.output.clone();
            }
            if dist > max_dist {
                max_dist = dist;
            }
        }
        // Evaluate the cone
        let mut top_sum = vec![0.0; self.output_space.len()];
        let mut bottom_sum = 0.0;
        for i in self.mappings.iter() {
            let dist = dist64(&in_value, &i.input).unwrap();
            let coef = ((max_dist - dist) * i.weight).powf(self.power);
            top_sum = vector_add(&top_sum, &vector_multiply(&i.output, &coef)).unwrap();
            bottom_sum += coef;
        }
        vector_multiply(&top_sum, &(1.0 / bottom_sum))
    }
}

// Tests

#[cfg(test)]
mod tests {
    // use super::*;
    #[test]
    fn test() {}
}
