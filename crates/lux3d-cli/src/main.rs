use clap::Parser;
use lux3d_cli::{
    inspect_model, normalize_weights, quantize_weights, run_model, run_ovox, Cli, Command,
    WeightsCommand,
};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => {
            let rendered = inspect_model(args.repo_root, args.family.model_family())?;
            println!("{rendered}");
        }
        Command::Weights(args) => match args.command {
            WeightsCommand::Normalize(family) => {
                let rendered = normalize_weights(
                    family.repo_root,
                    family.family.model_family(),
                    family.raw_model_dir,
                    family.output_dir,
                )?;
                println!("{rendered}");
            }
            WeightsCommand::Quantize(args) => println!("{}", quantize_weights(args)?),
        },
        Command::Run(args) => println!("{}", run_model(args)?.display()),
        Command::Ovox(args) => println!("{}", run_ovox(args)?),
    }

    Ok(())
}
