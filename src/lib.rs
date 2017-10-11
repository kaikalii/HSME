extern crate rand;

use rand::Rng;
use std::cmp;

// Utility

// Returns the max of two numbers (of any type)
// Why this is not std::cmp by default is beyond me.
pub fn max<T>(a: T, b: T) -> T
    where T: cmp::PartialOrd
{
    if a < b {
        return b;
    }
    a
}

// Adds two vectors and returns their sum
fn vector_add(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    if a.len() != b.len() {
        panic!("The Vecs' length must be equal for them to be added!");
    }
    let mut result: Vec<f64> = Vec::new();
    for (i, &n) in a.iter().enumerate() {
        result.push(n + b[i]);
    }
    result
}

// Multiplies each element in a vector by a constant and returns the product vector
fn vector_multiply(v: &Vec<f64>, c: &f64) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();
    for i in v.iter() {
        result.push(i * c);
    }
    result
}

// Returns the euclidean distance between two Vec<64>s
fn dist64(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("The Vecs' length must be equal to find their distance!");
    }
    let mut sum = 0.0;
    for (i, &n) in a.iter().enumerate() {
        sum += (n - b[i]).powf(2.0);
    }
    sum.powf(0.5)
}

// Space

// A simply struct that defines the acceptable bounds of a space of an arbitrary numbers of dimensions

#[derive(Debug)]
#[derive(Clone)]
pub struct Space {
    bounds: Vec<(f64, f64)>,
}

impl Space {

    // Creates a new Space with empty bounds
    pub fn new() -> Space {
        let s = Space {
            bounds: Vec::new(),
        };
        s
    }
    // Consumes a space and sets its bounds
    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Space {
        self.bounds = bounds;
        self
    }
    // Returns the number of dimensions of the space
    pub fn order(&self) -> usize {
        self.bounds.len()
    }
    // Adds a new bound to the space
    pub fn add_bound(&mut self, bound: (f64, f64)) {
        self.bounds.push(bound);
    }
}

// Mapping

// A struct for defining a mapping of a point in space A to a point in space B. Spaces A and B may or may not have the same number of dimensions. The mapping has a weight that is used for conversion calculations

#[derive(Debug)]
#[derive(Clone)]
pub struct Mapping {
    input: Vec<f64>,
    output: Vec<f64>,
    weight: f64,
}

impl Mapping {
    // Create a new empty mapping with half weight
    pub fn new() -> Mapping {
        let map = Mapping {
            input: Vec::new(),
            output: Vec::new(),
            weight: 0.5,
        };
        map
    }
    // Consume a mapping and set its points
    pub fn with_points(mut self, input: Vec<f64>, output: Vec<f64>) -> Mapping {
        self.input = input;
        self.output = output;
        self
    }
    // Consume a mapping and set its weight
    pub fn with_weight(mut self, weight: f64) -> Mapping {
        self.weight = weight;
        self
    }
}

// IntermapFunction

// An enum for defining an intermap function - a function that determines the way the output of a conversion is calculated when the input is not among the inputs of the conversion's mappings

// Conversion

// A struct that defines a conversion between two spaces. It is essentially a collection of mappings, all of who's inputs and outputs fall within the bounds of the conversion's spaces. The conversion can evaluate arbitrary inputs for their output based on the conversion's mappings and its intermap function.

#[derive(Debug)]
#[derive(Clone)]
pub struct Conversion {
    input: Space,
    output: Space,
    mappings: Vec<Mapping>,
    power: f64,
}

impl Conversion {
    // Creates a new conversion from input and output spaces. The conversion is initialized with a single averaged mapping.
    pub fn from_io(input: Space, output: Space) -> Conversion {
        let mut conv = Conversion {
            input,
            output,
            mappings: Vec::new(),
            power: 2.0,
        };
        let mut inpoint: Vec<f64> = Vec::new();
        for i in conv.input.bounds.iter() {
            inpoint.push((i.0 + i.1)/2.0);
        }
        let mut outpoint: Vec<f64> = Vec::new();
        for i in conv.output.bounds.iter() {
            outpoint.push((i.0 + i.1)/2.0);
        }
        let avg_mapping = Mapping::new().with_points(inpoint, outpoint);
        conv.mappings.push(avg_mapping);
        conv
    }
    // Sets the power
    pub fn with_power(mut self, power: f64) -> Conversion {
        self.power = power;
        self
    }
    // Removes the default mapping
    pub fn without_default_mapping(mut self) -> Conversion {
        if self.mappings.len() == 1 {
            self.mappings.pop();
        }
        self
    }
    // Adds a user-defined mapping to the conevrsion
    pub fn add_mapping(&mut self, m: Mapping) {
        if m.input.len() != self.input.order() {
            panic!("The mapping's input size is not equal to the conversion's input size!");
        }
        if m.output.len() != self.output.order() {
            panic!("The mapping's output size is not equal to the conversion's output size!");
        }
        self.mappings.push(m);
    }
    // Adds a random mapping to the conversion. This mapping will have points that fall within the conversion's space's bounds.
    pub fn add_random_mapping(&mut self) {
        let mut inpoint: Vec<f64> = Vec::new();
        let mut outpoint: Vec<f64> = Vec::new();
        let mut rng = rand::thread_rng();

        for &(min, max) in self.input.bounds.iter() {
            let range = max - min;
            inpoint.push(rng.gen_range(0.0, range) + min);
        }

        for &(min, max) in self.output.bounds.iter() {
            let range = max - min;
            outpoint.push(rng.gen_range(0.0, range) + min);
        }

        self.mappings.push(Mapping::new().with_points(inpoint, outpoint));
    }
    // Evaluates an inputs vector and returns the optional output based on the conversion's mappings and its intermap function.
    pub fn convert(&self, in_value: Vec<f64>) -> Option<Vec<f64> > {
        // find the maximum distance
        let mut max_dist = 0.0;
        for i in self.mappings.iter() {
            let dist = dist64(&in_value, &i.input);
            if dist == 0.0 {
                return Some(i.output.clone());
            }
            if dist > max_dist {
                max_dist = dist;
            }
        }
        // Evaluate the cone
        let mut top_sum = vec![0.0; self.output.bounds.len()];
        let mut bottom_sum = 0.0;
        for i in self.mappings.iter() {
            let dist = dist64(&in_value, &i.input);
            let coef = ((max_dist - dist) * i.weight).powf(self.power);
            top_sum = vector_add(&top_sum, &vector_multiply(&i.output, &coef));
            bottom_sum += coef;
        }
        let result = vector_multiply(&top_sum, &(1.0/bottom_sum));
        Some(result)
    }
}

// Pipeline

// A struct that defines a series of conversions. The order of conversion N's output always equals the order of conversion (N+1)'s input.

#[derive(Debug)]
#[derive(Clone)]
pub struct Pipeline {
    conversions: Vec<Conversion>,
}

impl Pipeline {
    // Contructs a Pipeline with a single Conversion which has only one mapping
    pub fn from_io(input: Space, output: Space) -> Pipeline {
        let pipe = Pipeline {
            conversions: vec![Conversion::from_io(input, output)],
        };
        pipe
    }
    // Runs an input vector through every conversion and return the final output
    pub fn convert(&self, in_value: Vec<f64>) -> Option<Vec<f64> > {
        let mut data = Some(in_value);
        for i in self.conversions.iter() {
            if let Some(d) = data {
                data = i.convert(d);
            }
        }
        data
    }
    // Mutates by adding a random mapping in a random conversion
    fn mutate_add_mapping(&mut self) {
        let mut rng = rand::thread_rng();
        let conv_index = rng.gen::<usize>() % self.conversions.len();
        self.conversions[conv_index].add_random_mapping();
    }
    // Mutates by changing either the input or output of a random mapping in a random conversion
    fn mutate_change_mapping(&mut self) {
        let mut rng = rand::thread_rng();
        let conv_index = rng.gen::<usize>() % self.conversions.len();
        let mapping_index = rng.gen::<usize>() % self.conversions[conv_index].mappings.len();
        let i_or_o = rng.gen::<usize>() % 2;
        if i_or_o == 0 {
            let mut new_vec: Vec<f64> = Vec::new();
            for i in self.conversions[conv_index].input.bounds.iter() {
                let new_val = rng.gen_range(i.0, i.1);
                new_vec.push(new_val);
            }
            self.conversions[conv_index].mappings[mapping_index].input = new_vec;
        }
        else {
            let mut new_vec: Vec<f64> = Vec::new();
            for i in self.conversions[conv_index].output.bounds.iter() {
                let new_val = rng.gen_range(i.0, i.1);
                new_vec.push(new_val);
            }
            self.conversions[conv_index].mappings[mapping_index].output = new_vec;
        }
    }
    // Mutates by changing the weight of a random mapping in a random conversion to a random number
    fn mutate_change_weight(&mut self) {
        let mut rng = rand::thread_rng();
        let conv_index = rng.gen::<usize>() % self.conversions.len();
        let mapping_index = rng.gen::<usize>() % self.conversions[conv_index].mappings.len();
        self.conversions[conv_index].mappings[mapping_index].weight = rng.gen::<f64>() % 1.0;
    }
    // Mutates by adding a new conversion at the end
    fn mutate_add_conversion(&mut self) {
        let ending_bounds: Vec<(f64, f64)>;
        if let Some(last) = self.conversions.last() {
            ending_bounds = last.output.bounds.clone();
        }
        else {
            panic!("Tried to add an ending conversion to an empty pipeline");
        }
        let new_start_space = Space::new().with_bounds(ending_bounds.clone());
        let new_end_space = Space::new().with_bounds(ending_bounds);
        let mut conv = Conversion::from_io(new_start_space, new_end_space);
        if let Some(last) = self.conversions.last() {
            for m in last.mappings.iter() {
                conv.add_mapping(Mapping::new().with_points(m.output.clone(), m.output.clone()));
            }
        }
        self.conversions.push(conv);
    }
    // Mutates by adding a dimension to a space shared by two conversion
    fn mutate_add_dimension(&mut self) -> bool {
        if self.conversions.len() < 2 {
            return false;
        }
        let mut rng = rand::thread_rng();
        let mut dim_count_sum = 0.0;
        for i in 0..(self.conversions.len() - 1) {
            dim_count_sum += 1.0/(self.conversions[i].output.bounds.len() as f64);
        }
        let mut lower_index = 0;
        let mut rnd = rng.gen_range(0.0, dim_count_sum);
        for (i, n) in self.conversions.iter().enumerate() {
            if rnd < 1.0/(n.output.bounds.len() as f64) {
                lower_index = i;
                break;
            }
            rnd -= 1.0/(n.output.bounds.len() as f64);
        }
        let mut new_bound = (0.0, 0.0);
        for i in self.conversions[lower_index].output.bounds.iter() {
            if i.0 < new_bound.0 {
                new_bound.0 = i.0;
            }
            if i.1 > new_bound.1 {
                new_bound.1 = i.1;
            }
        }
        self.conversions[lower_index].output.add_bound(new_bound.clone());
        self.conversions[lower_index + 1].input.add_bound(new_bound.clone());
        let new_val = (new_bound.0 + new_bound.1) / 2.0;
        for i in self.conversions[lower_index].mappings.iter_mut() {
            i.output.push(new_val);
        }
        for i in self.conversions[lower_index + 1].mappings.iter_mut() {
            i.input.push(new_val);
        }
        true
    }
    // Chooses a random mutation type to execute based on some weights
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
    // Returns a vector of the number of dimensions at each stage of the pipeline
    pub fn get_space_dims(&self) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::new();
        for i in self.conversions.iter() {
            result.push(i.input.bounds.len());
        }
        result.push(self.conversions[self.conversions.len() - 1].output.bounds.len());
        result
    }
    // Returns a vector of the number of mappings in each conversion
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
    #[test]
    fn it_works() {
    }
}
