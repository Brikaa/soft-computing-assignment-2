use rand::{thread_rng, Rng};
use text_io::scan;

type Chromosome = Vec<f64>;
struct Solution {
    chromosome: Chromosome,
    fitness: f64,
}

struct Point {
    x: f64,
    y: f64,
}

fn create_initialize_population(
    lb: f64,
    ub: f64,
    degree: u32,
    pop_size: u32,
) -> impl Fn() -> Vec<Chromosome> {
    move || {
        let mut population: Vec<Chromosome> = Vec::new();
        let mut rng = thread_rng();
        for _ in 1..=pop_size {
            let mut chromosome: Vec<f64> = Vec::new();
            for _ in 1..=degree + 1 {
                chromosome.push(rng.gen_range(lb..=ub));
            }
            population.push(chromosome);
        }
        population
    }
}

fn create_evaluate_fitness(points: &Vec<Point>) -> impl Fn(&Chromosome) -> f64 + '_ {
    move |chromosome| {
        let mut total_deviation = 0_f64;
        for point in points {
            let mut y = 0_f64;
            for i in 0..chromosome.len() {
                y += chromosome[i] * point.x.powi(i.try_into().unwrap());
            }
            total_deviation += (y - point.y).powi(2);
        }
        total_deviation / (points.len() as f64)
    }
}

fn create_tournament_selection(
    no_tournaments: u32,
) -> impl for<'a> Fn(&Vec<&'a Solution>) -> Vec<&'a Solution> {
    move |solutions| {
        let mut selected: Vec<&Solution> = Vec::new();
        let mut rng = thread_rng();
        for _ in 1..=no_tournaments {
            let r1 = rng.gen_range(0..solutions.len());
            let mut r2 = r1;
            while r2 == r1 {
                r2 = rng.gen_range(0..solutions.len());
            }
            let first = solutions[r1];
            let second = solutions[r2];
            let best = if first.fitness < second.fitness {
                first
            } else {
                second
            };
            selected.push(best);
        }
        selected
    }
}

fn create_crossover(
    crossover_rate: f64,
    no_points: u32,
) -> impl Fn(&Chromosome, &Chromosome) -> (Chromosome, Chromosome) {
    move |parent1, parent2| {
        let mut rng = thread_rng();
        if rng.gen::<f64>() > crossover_rate {
            return (parent1.clone(), parent2.clone());
        }
        let l = parent1.len();
        let mut offspring1: Chromosome = Vec::new();
        let mut offspring2: Chromosome = Vec::new();
        let mut points = Vec::new();
        for _ in 1..=no_points {
            let p = rng.gen_range(1..l);
            points.push(p);
        }
        points.sort();
        let mut prev = 0;
        let mut a = parent1;
        let mut b = parent2;
        for i in 1..=points.len() {
            let p = points[i - 1];
            for j in (prev + 1)..=p {
                offspring1.push(a[j - 1]);
                offspring2.push(b[j - 1]);
                let temp = a;
                a = b;
                b = temp;
            }
            prev = p;
        }
        for i in prev + 1..=l {
            offspring1.push(a[i - 1]);
            offspring2.push(b[i - 1]);
        }
        (offspring1, offspring2)
    }
}

fn create_mutation(
    dependency_factor: f64,
    lb: f64,
    ub: f64,
    mutation_rate: f64,
    max_gen: u32,
) -> impl Fn(u32, &mut Chromosome) {
    move |current_gen, chromosome| {
        let mut rng = thread_rng();
        let l = chromosome.len();
        for i in 0..l {
            if rng.gen::<f64>() <= mutation_rate {
                let ri1 = rng.gen::<f64>();
                let left = ri1 <= 0.5;
                let y = if left {
                    chromosome[i] - lb
                } else {
                    ub - chromosome[i]
                };
                let delta = y
                    * (1_f64
                        - rng
                            .gen::<f64>()
                            .powf(1_f64 - current_gen as f64 / max_gen as f64)
                            .powf(dependency_factor));
                if left {
                    chromosome[i] -= delta;
                } else {
                    chromosome[i] += delta;
                }
            }
        }
    }
}

fn create_create_solutions(
    evaluate_fitness: impl Fn(&Chromosome) -> f64,
) -> impl Fn(Vec<Chromosome>) -> Vec<Solution> {
    move |chromosomes| {
        let mut result: Vec<Solution> = Vec::new();
        for chromosome in chromosomes {
            let fitness = evaluate_fitness(&chromosome);
            result.push(Solution {
                chromosome,
                fitness,
            });
        }
        result
    }
}

fn decode_chromosome(chromosome: &Chromosome) -> String {
    let mut result = String::new();
    for i in (0..chromosome.len()).rev() {
        if i == 0 {
            result += &format!("{}", chromosome[i]);
        } else {
            result += &format!("{}x^{}", chromosome[i], i);
        }
        if i != 0 {
            result += " + ";
        }
    }
    result
}

fn run_genetic_algorithm(
    initialize_population: impl Fn() -> Vec<Chromosome>,
    create_solutions: impl Fn(Vec<Chromosome>) -> Vec<Solution>,
    crossover: impl Fn(&Chromosome, &Chromosome) -> (Chromosome, Chromosome),
    mutation: impl Fn(u32, &mut Chromosome),
    max_gen: u32,
    select: impl for<'a> Fn(&Vec<&'a Solution>) -> Vec<&'a Solution>,
    k: u32,
    no_elites: u32,
    current_test_case: u32,
) -> Solution {
    let population = initialize_population();
    let mut current_solutions = create_solutions(population);
    let comparison_fn = |a: &Solution, b: &Solution| b.fitness.partial_cmp(&a.fitness).unwrap();
    current_solutions.sort_by(comparison_fn);
    for current_gen in 1..=max_gen {
        let mut solution_refs = Vec::new();
        for sol in &current_solutions {
            solution_refs.push(sol);
        }
        let selected = select(&solution_refs);
        let mut offsprings: Vec<Chromosome> = Vec::new();
        for i in (0..selected.len()).step_by(2) {
            let (offspring1, offspring2) =
                crossover(&selected[i].chromosome, &selected[i + 1].chromosome);
            offsprings.push(offspring1);
            offsprings.push(offspring2);
        }

        let offset = offsprings.len() as u32 - k;
        for _ in 1..=offset {
            offsprings.pop().unwrap();
        }
        for offspring in &mut offsprings {
            mutation(current_gen, offspring);
        }
        let mut elites: Vec<Solution> = Vec::new();
        for _ in 1..=no_elites {
            let solution = current_solutions.pop().unwrap();
            elites.push(solution);
        }
        current_solutions = create_solutions(offsprings);
        current_solutions.append(&mut elites);
        current_solutions.sort_by(comparison_fn);

        let best = &current_solutions[current_solutions.len() - 1];
        eprintln!(
            "Test case: {}
Current generation: {}
Best chromosome {:?}
Best fitness: {}",
            current_test_case, current_gen, best.chromosome, best.fitness
        );
    }
    current_solutions.pop().unwrap()
}

fn main() {
    let lb = -10_f64;
    let ub = 10_f64;
    let pop_size = 800;
    let max_gen = 400;
    let k = (0.95 * pop_size as f64).ceil() as u32;
    let no_elites = pop_size - k;
    let crossover_rate = 0.9;
    let mutation_rate = 0.03;
    let dependency_factor = 2.5;
    let no_test_cases: u32;
    scan!("{}", no_test_cases);
    for current_test_case in 1..=no_test_cases {
        let (no_points, degree): (u32, u32);
        scan!("{} {}", no_points, degree);
        let mut points = Vec::new();
        for _ in 1..=no_points {
            let (x, y): (f64, f64);
            scan!("{} {}", x, y);
            points.push(Point { x, y })
        }
        let initialize_population = create_initialize_population(lb, ub, degree, pop_size);
        let evaluate_fitness = create_evaluate_fitness(&points);
        let create_solutions = create_create_solutions(evaluate_fitness);
        let select = create_tournament_selection(((k as f64 / 2.0).ceil() as u32) * 2);
        let crossover = create_crossover(crossover_rate, 2);
        let mutation = create_mutation(dependency_factor, lb, ub, mutation_rate, max_gen);
        let solution = run_genetic_algorithm(
            initialize_population,
            create_solutions,
            crossover,
            mutation,
            max_gen,
            select,
            k,
            no_elites,
            current_test_case,
        );
        println!(
            "Test case: {}
Best solution: {}
Mean square error: {}",
            current_test_case,
            decode_chromosome(&solution.chromosome),
            solution.fitness
        )
    }
}
