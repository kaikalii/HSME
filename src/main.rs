extern crate hsm;
extern crate rand;
extern crate serde;
extern crate serde_json;

use std::{collections::HashSet, env, fs::File, io::Read, path::PathBuf};

use hsm::*;

use rand::{thread_rng, Rng};

fn main() {
    let mut rng = thread_rng();
    let mut args = env::args();
    args.next();
    let mut file_stem_set = false;
    let mut file_stem;
    let mut json_file = PathBuf::new();
    let mut data_file = PathBuf::new();
    let mut txt_file = PathBuf::new();
    let mut build = false;
    let mut test = false;
    let mut eval = false;
    let mut ratio = 0.2;
    let mut nosave = false;
    let mut power = 2.0;
    let mut discrete_output = false;
    while let Some(arg) = args.next() {
        match arg.as_ref() {
            "-b" | "--build" => build = true,
            "-t" | "--test" => test = true,
            "-e" | "--eval" => eval = true,
            "-d" | "--discrete" => discrete_output = true,
            "-r" | "--ratio" => {
                ratio = args.next()
                    .map(|x| x.parse::<f64>().expect("Invalid test ratio"))
                    .unwrap_or(0.2)
            }

            "-p" | "--power" => {
                power = args.next()
                    .map(|x| x.parse::<f64>().expect("Invalid hsm power"))
                    .unwrap_or(2.0)
            }
            "--nosave" => nosave = true,
            "-h" | "--help" => {
                println!(
                    "
usage:
    decision-tree [data name] <flags>

flags:
    -b | --build        Builds the tree from a subset of the data
    -t | --test         Tests the tree on a subset of the data
    -e | --eval         Evaluate news data
    -r | --ratio        Sets the ratio of training data to total data
    -p | --power        Serts the hyperspace map power
    -d | --discrete     Set the output value as discrete
    --nosave            Prevents the map from saving to a file
    -h | --help         Prints this message
                "
                );
                return;
            }
            _ => {
                file_stem = arg;
                json_file = PathBuf::from(file_stem.clone()).with_extension("json");
                data_file = PathBuf::from(file_stem.clone()).with_extension("data");
                txt_file = PathBuf::from(file_stem.clone()).with_extension("txt");
                file_stem_set = true;
            }
        }
    }

    if !file_stem_set {
        println!("Expected data file path stem");
        return;
    }

    // Load the data
    let mut data_bytes = Vec::new();
    File::open(data_file)
        .expect("unable to open data file")
        .read_to_end(&mut data_bytes)
        .expect("Unable to read data file to string");
    let full_string = String::from_utf8_lossy(&data_bytes);
    let data_strings: Vec<String> = full_string
        .split_whitespace()
        .map(|x| x.to_string())
        .collect();
    let count = (data_strings.len() as f64 * ratio) as usize;

    // Read the data from the file into a vector
    let data: Vec<(Vec<f64>, f64)> = data_strings
        .into_iter()
        .map(|entry| {
            let mut attributes: Vec<f64> = entry
                .split(",")
                .map(|x| {
                    x.parse()
                        .expect(&format!("{} cannot be parse into an f64", x))
                })
                .collect();
            let outcome = attributes.pop().unwrap();
            (attributes, outcome)
        })
        .collect();

    // Load a subset of the data
    let mut training_data: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut used_entries = HashSet::new();
    while used_entries.len() < count {
        let random_entry = rng.gen_range(0, data.len());
        if !used_entries.contains(&random_entry) {
            training_data.push(data[random_entry].clone());
            used_entries.insert(random_entry);
        }
    }
    println!("training data size: {}", training_data.len());
    let mut test_data: Vec<(Vec<f64>, f64)> = Vec::new();
    for (i, entry) in data.iter().enumerate() {
        if !used_entries.contains(&i) {
            test_data.push(entry.clone());
        }
    }
    println!("test data size: {}", test_data.len());

    // Determine the mins a maxes of each attribute

    let mut mins = training_data.first().expect("no data").clone();
    let mut maxs = mins.clone();
    for (inputs, output) in &training_data {
        for (input, (min, max)) in inputs.iter().zip(mins.0.iter_mut().zip(maxs.0.iter_mut())) {
            *min = min.min(*input);
            *max = max.max(*input);
        }
        mins.1 = mins.1.min(*output);
        maxs.1 = maxs.1.max(*output);
    }

    // Building a hyperspace map
    let mut conv = None;
    if build {
        println!("Building...");

        // Create the conversion
        conv = Some(
            Conversion::from_io_spaces(
                mins.0.into_iter().zip(maxs.0.into_iter()).collect(),
                vec![(mins.1, maxs.1)],
            ).with_power(power),
        );

        for (inputs, output) in training_data {
            conv.as_mut()
                .unwrap()
                .add_mapping(Mapping::from_points(inputs, vec![output]));
        }

        println!("Hyperspace map construction complete");

        // Save the hsm
        if !nosave {
            let out_file = File::create(json_file.clone()).expect("Unable to create output file");
            serde_json::to_writer_pretty(out_file, &conv).expect("Unable to serialize hsm");
            println!("Hyperspace map saved to file");
        }
    }
    // Build the hsm from the hsm file if it was not built
    if conv.is_none() {
        conv = Some(
            serde_json::from_reader(
                File::open(json_file.clone()).expect("unable to open hsm file"),
            ).expect("Unable to deserialize hsm file"),
        );
    }
    // Testing a hyperspace map
    if test {
        // Test the test data
        let mut successes = 0;
        let mut failures = 0;
        let avg_error = test_data.iter().fold(0.0, |sum, entry| {
            sum + (conv.as_ref()
                .unwrap()
                .convert(entry.0.clone())
                .first()
                .expect("convert returned empty vector") - entry.1)
                .abs()
        }) / test_data.len() as f64;

        for entry in &test_data {
            if conv.as_ref()
                .unwrap()
                .convert(entry.0.clone())
                .first()
                .expect("convert returned empty vector")
                .floor() == entry.1.floor()
            {
                successes += 1;
            } else {
                failures += 1;
            }
        }

        // Report results
        println!("..............................");
        if discrete_output {
            println!("{} successes, {} failures", successes, failures);
            println!(
                "{}% accuracy",
                successes as f32 / (successes + failures) as f32 * 100.0
            );
        } else {
            println!(
                "Test data conversion has an average error of {:.4} in a range of {} ({} - {})",
                avg_error,
                maxs.1 - mins.1,
                mins.1,
                maxs.1
            );
        }
        println!("..............................");
    }
    // Evaluate new data
    if eval {
        // Load the eval data
        let mut eval_data_bytes = Vec::new();
        File::open(txt_file)
            .expect("unable to open eval file")
            .read_to_end(&mut eval_data_bytes)
            .expect("Unable to read eval file to string");
        let eval_full_string = String::from_utf8_lossy(&eval_data_bytes);
        let eval_data_strings: Vec<String> = eval_full_string
            .split_whitespace()
            .map(|x| x.to_string())
            .collect();

        // Read the eval data from the file into a vector
        let eval_data: Vec<Vec<f64>> = eval_data_strings
            .into_iter()
            .map(|entry| {
                entry
                    .split(",")
                    .map(|x| x.parse().expect("Unable to parse eval data into f64"))
                    .collect()
            })
            .collect();

        // Evaluate the data
        println!("Evaluation:");
        for entry in eval_data {
            let result = conv.as_ref().unwrap().convert(entry.clone());
            println!("{:?} : {:?}", entry, result);
        }
        println!("..............................");
    }
}
