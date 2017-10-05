extern crate rand;

use rand::Rng;

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

fn vector_multiply(v: &Vec<f64>, c: &f64) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();
    for i in v.iter() {
        result.push(i * c);
    }
    result
}

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

#[derive(Debug)]
#[derive(Clone)]
pub struct Space {
    bounds: Vec<(f64, f64)>,
}

impl Space {
    pub fn new() -> Space {
        let s = Space {
            bounds: Vec::new(),
        };
        s
    }
    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Space {
        self.bounds = bounds;
        self
    }
    pub fn order(&self) -> usize {
        self.bounds.len()
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Mapping {
    input: Vec<f64>,
    output: Vec<f64>,
    weight: f64,
}

impl Mapping {
    pub fn new() -> Mapping {
        let map = Mapping {
            input: Vec::new(),
            output: Vec::new(),
            weight: 0.5,
        };
        map
    }
    pub fn with_points(mut self, input: Vec<f64>, output: Vec<f64>) -> Mapping {
        self.input = input;
        self.output = output;
        self
    }
    pub fn with_weight(mut self, weight: f64) -> Mapping {
        self.weight = weight;
        self
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub enum IntermapFunction {
    Nearest,
    InverseDistance{ power: f64, range: Option<f64> },
    AverageInRange(f64),
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Conversion {
    input: Space,
    output: Space,
    mappings: Vec<Mapping>,
    intermap_function: IntermapFunction,
}

impl Conversion {
    pub fn from_io(input: Space, output: Space) -> Conversion {
        let conv = Conversion {
            input,
            output,
            mappings: Vec::new(),
            intermap_function: IntermapFunction::InverseDistance{ power: 2.0, range: None},
        };
        conv
    }
    pub fn with_itermap_function(mut self, imf: IntermapFunction) -> Conversion {
        self.intermap_function = imf;
        self
    }
    pub fn add_mapping(&mut self, m: Mapping) {
        if m.input.len() != self.input.order() {
            panic!("The mapping's input size is not equal to the conversion's input size!");
        }
        if m.output.len() != self.output.order() {
            panic!("The mapping's output size is not equal to the conversion's output size!");
        }
        self.mappings.push(m);
    }
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
                        vector_sum = vector_add(&vector_sum, &m.output);
                        count += 1.0;
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




#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
