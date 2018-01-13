extern crate rand;
#[macro_use]
extern crate getset;

use rand::Rng;
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
        write!(f, "Attempted to perform an operation using two vectors with different orders.")
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

#[derive(Debug, Clone, PartialEq, Getters, Setters)]
pub struct Mapping {
    input: Vec<f64>,
    output: Vec<f64>,
    weight: f64,
}

impl Mapping {
    /// Creates a new mapping that connects the centers of two spaces
    pub fn avg_from_spaces(input_space: &Vec<(f64,f64)>, output_space: &Vec<(f64,f64)>) -> Mapping {
        let mut inpoint: Vec<f64> = Vec::new();
        let mut outpoint: Vec<f64> = Vec::new();

        for i in input_space.iter() {
            inpoint.push((i.0 + i.1)/2.0);
        }
        for i in output_space.iter() {
            outpoint.push((i.0 + i.1)/2.0);
        }

        let new_mapping = Mapping {
            input: inpoint,
            output: outpoint,
            weight: 0.5,
        };
        new_mapping
    }
    /// Creates a new mapping that connects the centers of two spaces
    pub fn random_from_spaces(input_space: &Vec<(f64,f64)>, output_space: &Vec<(f64,f64)>) -> Mapping {
        let mut inpoint: Vec<f64> = Vec::new();
        let mut outpoint: Vec<f64> = Vec::new();
        let mut rng = rand::thread_rng();

        for &(min, max) in input_space.iter() {
            let range = max - min;
            inpoint.push(rng.gen_range(0.0, range) + min);
        }

        for &(min, max) in output_space.iter() {
            let range = max - min;
            outpoint.push(rng.gen_range(0.0, range) + min);
        }
        let new_mapping = Mapping {
            input: inpoint,
            output: outpoint,
            weight: rng.gen_range(0.0, 1.0),
        };
        new_mapping
    }
    /// Initializes a mapping's input and output points
    fn from_points(input: &Vec<f64>, output: &Vec<f64>) -> Mapping {
        let new_mapping = Mapping {
            input: input.clone(),
            output: output.clone(),
            weight: 0.5,
        };
        new_mapping
    }
}

// Conversion

/// A struct that defines a conversion between two spaces. It is essentially a collection of mappings, all of whom's inputs and outputs fall within the bounds of the conversion's spaces. The conversion can evaluate arbitrary inputs for its output based on the conversion's mappings and its intermap function.

#[derive(Debug, Clone, PartialEq, Getters, Setters)]
pub struct Conversion {
    input_space: Vec<(f64,f64)>,
    output_space: Vec<(f64,f64)>,
    mappings: Vec<Mapping>,
    power: f64,
}

impl Conversion {
    /// Creates a new conversion from input and output spaces. The conversion is initialized with a single averaged mapping
    pub fn from_io(input_space: &Vec<(f64,f64)>, output_space: &Vec<(f64,f64)>) -> Conversion {
        let conv = Conversion {
            input_space: input_space.clone(),
            output_space: output_space.clone(),
            mappings: vec![Mapping::avg_from_spaces(&input_space, &output_space)],
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
    fn add_mapping(&mut self, m: Mapping) {
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
        let result = vector_multiply(&top_sum, &(1.0/bottom_sum));
        result
    }
}

// Pipeline

/// A struct that defines a series of conversions. The order of conversion N's output always equals the order of conversion (N+1)'s input.

#[derive(Debug, Clone, PartialEq, Getters, Setters)]
pub struct Pipeline {
    conversions: Vec<Conversion>,
}

impl Pipeline {
    /// Contructs a Pipeline with a single Conversion which has only one mapping
    pub fn from_io(input_space: &Vec<(f64,f64)>, output_space: &Vec<(f64,f64)>) -> Pipeline {
        let pipe = Pipeline {
            conversions: vec![Conversion::from_io(input_space, output_space)],
        };
        pipe
    }
    /// Runs an input vector through every conversion and returns the final output
    pub fn convert(&self, in_value: Vec<f64>) -> Vec<f64> {
        let mut data = in_value;
        for i in self.conversions.iter() {
                data = i.convert(data);
        }
        data
    }
    /// Mutates by adding a random mapping in a random conversion
    fn mutate_add_mapping(&mut self) {
        let mut rng = rand::thread_rng();
        let conv_index = rng.gen::<usize>() % self.conversions.len();
        let input_space = self.conversions[conv_index].input_space.clone();
        let output_space = self.conversions[conv_index].output_space.clone();
        self.conversions[conv_index].add_mapping(Mapping::random_from_spaces(&input_space, &output_space));
    }
    /// Mutates by changing either the input or output of a random mapping in a random conversion
    fn mutate_change_mapping(&mut self) {
        let mut rng = rand::thread_rng();
        let conv_index = rng.gen::<usize>() % self.conversions.len();
        let mapping_index = rng.gen::<usize>() % self.conversions[conv_index].mappings.len();
        let i_or_o = rng.gen::<usize>() % 2;
        if i_or_o == 0 {
            let mut new_vec: Vec<f64> = Vec::new();
            for i in self.conversions[conv_index].input_space.iter() {
                let new_val = rng.gen_range(i.0, i.1);
                new_vec.push(new_val);
            }
            self.conversions[conv_index].mappings[mapping_index].input = new_vec;
        }
        else {
            let mut new_vec: Vec<f64> = Vec::new();
            for i in self.conversions[conv_index].output_space.iter() {
                let new_val = rng.gen_range(i.0, i.1);
                new_vec.push(new_val);
            }
            self.conversions[conv_index].mappings[mapping_index].output = new_vec;
        }
    }
    /// Mutates by changing the weight of a random mapping in a random conversion to a random number
    fn mutate_change_weight(&mut self) {
        let mut rng = rand::thread_rng();
        let conv_index = rng.gen::<usize>() % self.conversions.len();
        let mapping_index = rng.gen::<usize>() % self.conversions[conv_index].mappings.len();
        self.conversions[conv_index].mappings[mapping_index].weight = rng.gen::<f64>() % 1.0;
    }
    /// Mutates by adding a new conversion at the end
    fn mutate_add_conversion(&mut self) {
        let ending_bounds: Vec<(f64, f64)>;
        if let Some(last) = self.conversions.last() {
            ending_bounds = last.output_space.clone();
        }
        else {
            panic!("Tried to add an ending conversion to an empty pipeline");
        }
        let new_start_space = ending_bounds.clone();
        let new_end_space = ending_bounds;
        let mut conv = Conversion::from_io(&new_start_space, &new_end_space);
        if let Some(last) = self.conversions.last() {
            for m in last.mappings.iter() {
                conv.add_mapping(Mapping::from_points(&m.output, &m.output));
            }
        }
        self.conversions.push(conv);
    }
    /// Mutates by adding a dimension to a space shared by two conversion
    fn mutate_add_dimension(&mut self) -> bool {
        if self.conversions.len() < 2 {
            return false;
        }
        let mut rng = rand::thread_rng();
        let mut dim_count_sum = 0.0;
        for i in 0..(self.conversions.len() - 1) {
            dim_count_sum += 1.0/(self.conversions[i].output_space.len() as f64);
        }
        let mut lower_index = 0;
        let mut rnd = rng.gen_range(0.0, dim_count_sum);
        for (i, n) in self.conversions.iter().enumerate() {
            if rnd < 1.0/(n.output_space.len() as f64) {
                lower_index = i;
                break;
            }
            rnd -= 1.0/(n.output_space.len() as f64);
        }
        let mut new_bound = (0.0, 0.0);
        for i in self.conversions[lower_index].output_space.iter() {
            if i.0 < new_bound.0 {
                new_bound.0 = i.0;
            }
            if i.1 > new_bound.1 {
                new_bound.1 = i.1;
            }
        }
        self.conversions[lower_index].output_space.push(new_bound.clone());
        self.conversions[lower_index + 1].input_space.push(new_bound.clone());
        let new_val = (new_bound.0 + new_bound.1) / 2.0;
        for i in self.conversions[lower_index].mappings.iter_mut() {
            i.output.push(new_val);
        }
        for i in self.conversions[lower_index + 1].mappings.iter_mut() {
            i.input.push(new_val);
        }
        true
    }
    /// Chooses a random mutation type to execute based on some weights
    pub fn mutate(&mut self) {
        let mutation_weights = vec![
            10.0, // mutate_change_mapping
            7.0, // mutate_change_weight
            1.0, // mutate_add_mapping
            0.4, // mutate_add_dimension
            0.1, // mutate_add_conversion
            ];
        let mut weight_sum = 0.0;
        for i in mutation_weights.iter() {
            weight_sum += *i;
        }
        let mut rng = rand::thread_rng();
        let mut chosen_index = rng.gen_range(0.0, weight_sum);
        let mut choice = 0;
        for (i, &n) in mutation_weights.iter().enumerate() {
            if chosen_index < n {
                choice = i;
                break;
            }
            chosen_index -= n;
        }
        match choice {
            0 => self.mutate_change_mapping(),
            1 => self.mutate_change_weight(),
            2 => self.mutate_add_mapping(),
            3 => if !self.mutate_add_dimension() {
                self.mutate();
            },
            4 => self.mutate_add_conversion(),
            _ => (),
        }
    }
    /// Returns a vector of the number of dimensions at each stage of the pipeline
    pub fn get_space_dims(&self) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::new();
        for i in self.conversions.iter() {
            result.push(i.input_space.len());
        }
        result.push(self.conversions[self.conversions.len() - 1].output_space.len());
        result
    }
    /// Returns a vector of the number of mappings in each conversion
    pub fn get_mapping_counts(&self) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::new();
        for i in self.conversions.iter() {
            result.push(i.mappings.len());
        }
        result.push(self.conversions[self.conversions.len() - 1].mappings.len());
        result
    }
}


// Tests

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        let mut pipe = Pipeline::from_io(
            &vec![(0.0,10.0);2],
            &vec![(0.0,100.0)],
        );
        println!("{:#?}", pipe);
        println!("Maps: {:?}", pipe.get_mapping_counts());
        println!("Dims: {:?}", pipe.get_space_dims());
        for i in 0..10 {
            println!("{}", i);
            pipe.mutate();
        }
        println!("{:#?}", pipe);
        println!("Maps: {:?}", pipe.get_mapping_counts());
        println!("Dims: {:?}", pipe.get_space_dims());
    }
}
