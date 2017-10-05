extern crate rand;

use rand::Rng;

#[derive(Debug)]
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
pub struct Conversion {
    input: Space,
    output: Space,
    mappings: Vec<Mapping>,
}

impl Conversion {
    pub fn from_io(input: Space, output: Space) -> Conversion {
        let conv = Conversion {
            input,
            output,
            mappings: Vec::new(),
        };
        conv
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
}




#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
