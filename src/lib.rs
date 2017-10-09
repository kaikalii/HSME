extern crate rand;

use rand::Rng;
use std::cmp;

// Utility

// Resturns the max of two numbers (of any type)
// Why this is not std::cmp by default is beyond me.
fn max<T>(a: T, b: T) -> T
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

#[derive(Debug)]
#[derive(Clone)]
// These functions map unkown inputs to the output of ...
pub enum IntermapFunction {
    // ... the nearest known input
    Nearest,
    // ... the sum of the outputs weighted by the inverse of their inputs' distance to the input in question. The inverse is raised to some power (2 or 3 seems to work best), and only inputs within some optional range are considered (lowering the range can help with performance).
    InverseDistance{ power: f64, range: Option<f64> },
    // ... the average of all outputs whose inputs are within some range
    AverageInRange(f64),
    // ... the average of all outputs whose inputs are within some range, weighted linearly by distance within that range.
    AverageInCone(f64),
}

// Conversion

// A struct that defines a conversion between two spaces. It is essentially a collection of mappings, all of who's inputs and outputs fall within the bounds of the conversion's spaces. The conversion can evaluate arbitrary inputs for their output based on the conversion's mappings and its intermap function.

#[derive(Debug)]
#[derive(Clone)]
pub struct Conversion {
    input: Space,
    output: Space,
    mappings: Vec<Mapping>,
    intermap_function: IntermapFunction,
}

impl Conversion {
    // Creates a new conversion from input and output spaces. The conversion is initialized with a single averaged mapping.
    pub fn from_io(input: Space, output: Space) -> Conversion {
        let mut conv = Conversion {
            input,
            output,
            mappings: Vec::new(),
            intermap_function: IntermapFunction::InverseDistance{power: 2.0, range: None}
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
    // Consumes a conversion and sets its intermap function
    pub fn with_itermap_function(mut self, imf: IntermapFunction) -> Conversion {
        self.intermap_function = imf;
        self
    }
    //
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
            inpoint.push(rng.gen::<f64>() % range + min);
        }

        for &(min, max) in self.output.bounds.iter() {
            let range = max - min;
            outpoint.push(rng.gen::<f64>() % range + min);
        }

        self.mappings.push(Mapping::new().with_points(inpoint, outpoint));
    }
    // Evaluates an inputs vector and returns the optional output based on the conversion's mappings and its intermap function.
    pub fn convert(&self, in_value: Vec<f64>) -> Option<Vec<f64> > {
        match self.intermap_function {
            IntermapFunction::InverseDistance{power, range} => {
                let mut vector_sum: Vec<f64> = vec![0.0;self.output.order()];
                let mut denom_sum = 0.0;
                for m in self.mappings.iter() {
                    let dist = dist64(&m.input, &in_value);
                    if dist == 0.0 {
                        let result = m.output.clone();
                        return Some(result);
                    }
                    let weight_dist = m.weight * (1.0 / dist).powf(power);
                    let mut in_range = true;
                    if let Some(r) = range {
                        if dist > r {
                            in_range = false;
                        }
                    }
                    if in_range {
                        vector_sum = vector_add(&vector_sum, &vector_multiply(&m.output, &weight_dist));
                        denom_sum += weight_dist;
                    }
                }
                if denom_sum == 0.0 {
                    return None;
                }
                let result = vector_multiply(&vector_sum, &(1.0/denom_sum));
                Some(result)
            }
            IntermapFunction::Nearest => {
                let mut min_dist_mapping = (
                    dist64(&self.mappings[0].input, &in_value),
                    self.mappings[0].clone(),
                );
                for m in self.mappings.iter() {
                    let dist = dist64(&m.input, &in_value);
                    if dist < min_dist_mapping.0 {
                        min_dist_mapping = (dist, m.clone());
                    }
                }
                Some(min_dist_mapping.1.output)
            }
            IntermapFunction::AverageInRange(range) => {
                let mut vector_sum: Vec<f64> = vec![0.0;self.output.order()];
                let mut count = 0.0;
                for m in self.mappings.iter() {
                    let dist = dist64(&m.input, &in_value);
                    if dist == 0.0 {
                        let result = m.output.clone();
                        return Some(result);
                    }
                    let mut in_range = true;
                    if dist > range {
                        in_range = false;
                    }
                    if in_range {
                        vector_sum = vector_add(&vector_sum, &vector_multiply(&m.output, &m.weight));
                        count += m.weight;
                    }
                }
                if count == 0.0 {
                    return None;
                }
                let result = vector_multiply(&vector_sum, &(1.0/count));
                Some(result)
            }
            IntermapFunction::AverageInCone(range) => {
                let mut vector_sum: Vec<f64> = vec![0.0;self.output.order()];
                let mut count = 0.0;
                for m in self.mappings.iter() {
                    let dist = dist64(&m.input, &in_value);
                    if dist == 0.0 {
                        let result = m.output.clone();
                        return Some(result);
                    }
                    let mut in_range = true;
                    if dist > range {
                        in_range = false;
                    }
                    if in_range {
                        let multiplier = m.weight*max(0.0, (range - dist)/range);
                        vector_sum = vector_add(&vector_sum, &vector_multiply(&m.output, &multiplier));
                        count += multiplier;
                    }
                }
                if count == 0.0 {
                    return None;
                }
                let result = vector_multiply(&vector_sum, &(1.0/count));
                Some(result)
            }
        }

    }
}

// Pipeline

// A struct that defines a series of conversions.

pub struct Pipeline {
    conversions: Vec<Conversion>,
}

impl Pipeline {
    pub fn from_io(input: Space, output: Space) -> Pipeline {
        let pipe = Pipeline {
            conversions: vec![Conversion::from_io(input, output)],
        };
        pipe
    }
    pub fn convert(&self, in_value: Vec<f64>) -> Option<Vec<f64> > {
        let mut data = Some(in_value);
        for i in self.conversions.iter() {
            if let Some(d) = data {
                data = i.convert(d);
            }
        }
        data
    }
    fn mutate(&self) {
        
    }
}


// Tests

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
