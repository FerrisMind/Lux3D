use clap::Parser;
use lux3d_cli::{Cli, Command, DTypeKind, VggtPreprocessKind, run_model};
use lux3d_core::ModelFamily;

#[test]
fn run_command_accepts_model_path_and_cache_dir_without_repo_root() {
    let model_path = std::env::temp_dir().join("lux3d-models").join("pi3");
    let cache_dir = std::env::temp_dir().join("luxrt-cache");

    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "pi3".to_string(),
        "--source".to_string(),
        "input.mp4".to_string(),
        "--model-path".to_string(),
        model_path.display().to_string(),
        "--cache-dir".to_string(),
        cache_dir.display().to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };

    assert_eq!(Some(model_path), args.model_path);
    assert_eq!(Some(cache_dir), args.cache_dir);
}

#[test]
fn run_triposg_parses_family_and_option_overrides() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--steps".to_string(),
        "12".to_string(),
        "--guidance-scale".to_string(),
        "3.5".to_string(),
        "--seed".to_string(),
        "7".to_string(),
        "--num-tokens".to_string(),
        "1024".to_string(),
        "--dense-octree-depth".to_string(),
        "5".to_string(),
        "--hierarchical-octree-depth".to_string(),
        "6".to_string(),
        "--flash-decoder".to_string(),
        "false".to_string(),
        "--flash-octree-depth".to_string(),
        "7".to_string(),
        "--chunk-size".to_string(),
        "4096".to_string(),
        "--output".to_string(),
        "out.obj".to_string(),
    ]);

    assert_eq!("triposg", cli.command.family_name());
    assert_eq!(Some(ModelFamily::TripoSg), cli.command.family());

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(Some(12), args.steps);
    assert_eq!(Some(3.5), args.guidance_scale);
    assert_eq!(Some(7), args.seed);
    assert_eq!(Some(1024), args.num_tokens);
    assert_eq!(Some(5), args.dense_octree_depth);
    assert_eq!(Some(6), args.hierarchical_octree_depth);
    assert_eq!(Some(false), args.flash_decoder);
    assert_eq!(Some(7), args.flash_octree_depth);
    assert_eq!(Some(4096), args.chunk_size);
}

#[test]
fn run_triposg_flags_default_to_none() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--output".to_string(),
        "out.obj".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(None, args.steps);
    assert_eq!(None, args.guidance_scale);
    assert_eq!(None, args.seed);
    assert_eq!(None, args.num_tokens);
    assert_eq!(None, args.dense_octree_depth);
    assert_eq!(None, args.hierarchical_octree_depth);
    assert_eq!(None, args.flash_decoder);
    assert_eq!(None, args.flash_octree_depth);
    // New TripoSG core-option flags default to None (behavior unchanged).
    assert_eq!(None, args.faces);
    assert_eq!(None, args.dtype);
    assert_eq!(None, args.bounds);
    assert_eq!(None, args.timesteps);
}

/// The new TripoSG core options (`--faces` / `--dtype` / `--bounds` /
/// `--timesteps`) parse from the integration-test surface.
#[test]
fn run_triposg_parses_new_core_option_flags() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--faces".to_string(),
        "50000".to_string(),
        "--dtype".to_string(),
        "bf16".to_string(),
        "--bounds".to_string(),
        "-1.2,-1.3,-1.4,1.2,1.3,1.4".to_string(),
        "--timesteps".to_string(),
        "999,777,500,250,0".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(Some(50000), args.faces);
    assert_eq!(Some(DTypeKind::Bf16), args.dtype);
    assert_eq!(Some("-1.2,-1.3,-1.4,1.2,1.3,1.4".to_string()), args.bounds);
    assert_eq!(Some("999,777,500,250,0".to_string()), args.timesteps);
}

/// `--faces 0` parses (the documented "no simplification" value, the upstream
/// `-1` equivalent); negative face counts are not representable as usize and
/// are rejected by clap.
#[test]
fn run_triposg_faces_flag_accepts_zero_but_rejects_negative() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--faces".to_string(),
        "0".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);
    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(Some(0), args.faces);

    let parsed = Cli::try_parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--faces".to_string(),
        "-1".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);
    assert!(parsed.is_err(), "`--faces -1` must be rejected by clap");
}

/// Unknown `--dtype` values fail clap parsing with the possible values listed.
#[test]
fn run_triposg_dtype_flag_rejects_unknown_values() {
    let parsed = Cli::try_parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--dtype".to_string(),
        "q8_0".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);
    let error = parsed.expect_err("unknown dtype must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("invalid value") && message.contains("possible values"),
        "unexpected message: {message}"
    );
}

/// Dispatch-level regression (T5 review minor 3): `run_model` must reject a
/// TripoSG-only flag for another family BEFORE loading any weights, so a
/// plain `--device cpu` run fails fast with the rejection message and never
/// touches the source file or the canonical package.
#[test]
fn run_model_rejects_triposg_only_flag_at_dispatch_before_weights() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "pi3".to_string(),
        "--device".to_string(),
        "cpu".to_string(),
        "--source".to_string(),
        "does-not-exist.mp4".to_string(),
        "--steps".to_string(),
        "5".to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };

    let error = run_model(args).expect_err("`--steps` must be rejected for pi3 at dispatch");
    let message = error.to_string();
    assert!(
        message.contains("`--steps` is not supported for `pi3`"),
        "unexpected rejection message: {message}"
    );
}

#[test]
fn run_trellis2_parses_family_and_run_overrides() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "trellis2".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--model-path".to_string(),
        "models/TRELLIS.2-4B".to_string(),
        "--pipeline-type".to_string(),
        "1024_cascade".to_string(),
        "--steps".to_string(),
        "4".to_string(),
        "--seed".to_string(),
        "7".to_string(),
        "--max-num-tokens".to_string(),
        "8192".to_string(),
        "--texture".to_string(),
        "false".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);

    assert_eq!("trellis2", cli.command.family_name());
    assert_eq!(Some(ModelFamily::Trellis2), cli.command.family());

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(ModelFamily::Trellis2, args.family.model_family());
    assert_eq!(Some(4), args.steps);
    assert_eq!(Some(7), args.seed);
    assert_eq!(Some(8192), args.max_num_tokens);
    assert_eq!(Some(false), args.texture);
    assert_eq!(
        Some(lux3d_core::Trellis2PipelineType::Cascade1024),
        args.pipeline_type
            .map(lux3d_core::Trellis2PipelineType::from)
    );
}

#[test]
fn run_trellis2_parses_1536_cascade_and_per_stage_flags() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "trellis2".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--pipeline-type".to_string(),
        "1536_cascade".to_string(),
        "--ss-steps".to_string(),
        "6".to_string(),
        "--shape-steps".to_string(),
        "5".to_string(),
        "--tex-steps".to_string(),
        "4".to_string(),
        "--ss-guidance".to_string(),
        "4.5".to_string(),
        "--ss-rescale".to_string(),
        "0.3".to_string(),
        "--ss-interval".to_string(),
        "0.5,0.95".to_string(),
        "--num-samples".to_string(),
        "3".to_string(),
        "--return-latent".to_string(),
        "latents.npz".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);

    assert_eq!("trellis2", cli.command.family_name());
    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(
        Some(lux3d_core::Trellis2PipelineType::Cascade1536),
        args.pipeline_type
            .map(lux3d_core::Trellis2PipelineType::from)
    );
    assert_eq!(Some(6), args.ss_steps);
    assert_eq!(Some(5), args.shape_steps);
    assert_eq!(Some(4), args.tex_steps);
    assert_eq!(Some(4.5), args.ss_guidance);
    assert_eq!(Some(0.3), args.ss_rescale);
    assert_eq!(Some("0.5,0.95".to_string()), args.ss_interval);
    assert_eq!(Some(3), args.num_samples);
    assert_eq!(
        Some(std::path::PathBuf::from("latents.npz")),
        args.return_latent
    );
}

#[test]
fn run_trellis2_flags_default_to_none() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "trellis2".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--output".to_string(),
        "out.obj".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(None, args.pipeline_type);
    assert_eq!(None, args.max_num_tokens);
    assert_eq!(None, args.texture);
    // Per-stage sampler / num-samples / return-latent flags default to None.
    assert_eq!(None, args.ss_steps);
    assert_eq!(None, args.shape_steps);
    assert_eq!(None, args.tex_steps);
    assert_eq!(None, args.ss_guidance);
    assert_eq!(None, args.shape_guidance);
    assert_eq!(None, args.tex_guidance);
    assert_eq!(None, args.ss_rescale);
    assert_eq!(None, args.shape_rescale);
    assert_eq!(None, args.tex_rescale);
    assert_eq!(None, args.ss_interval);
    assert_eq!(None, args.shape_interval);
    assert_eq!(None, args.tex_interval);
    assert_eq!(None, args.num_samples);
    assert_eq!(None, args.return_latent);
    // Mesh post-process flags default to off/enabled (upstream to_glb
    // parameters come from the run options defaults).
    assert!(!args.no_postprocess);
    assert_eq!(None, args.decimation_target);
    assert_eq!(None, args.fill_hole_perimeter);
    assert_eq!(None, args.min_component_area);
}

#[test]
fn run_trellis2_parses_postprocess_flags() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "trellis2".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--no-postprocess".to_string(),
        "--decimation-target".to_string(),
        "4096".to_string(),
        "--fill-hole-perimeter".to_string(),
        "0.05".to_string(),
        "--min-component-area".to_string(),
        "1e-4".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert!(args.no_postprocess);
    assert_eq!(Some(4096), args.decimation_target);
    assert_eq!(Some(0.05), args.fill_hole_perimeter);
    assert_eq!(Some(1e-4), args.min_component_area);
}

/// TRELLIS.2 must reject foreign-family flags at dispatch BEFORE loading any
/// weights, mirroring the TripoSG fail-fast behavior.
#[test]
fn run_model_rejects_trellis2_foreign_flag_at_dispatch_before_weights() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "trellis2".to_string(),
        "--device".to_string(),
        "cpu".to_string(),
        "--source".to_string(),
        "does-not-exist.png".to_string(),
        "--mc-resolution".to_string(),
        "256".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };

    let error = run_model(args).expect_err("`--mc-resolution` must be rejected for trellis2");
    let message = error.to_string();
    assert!(
        message.contains("`--mc-resolution` is not supported for `trellis2`"),
        "unexpected rejection message: {message}"
    );
}

/// The TRELLIS.2 textured-GLB export flags parse with their values intact.
#[test]
fn run_trellis2_parses_texture_export_flags() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "trellis2".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--texture-size".to_string(),
        "1024".to_string(),
        "--no-texture-export".to_string(),
        "--no-uv-unwrap".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);

    assert_eq!("trellis2", cli.command.family_name());
    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(Some(1024), args.texture_size);
    assert!(args.no_texture_export);
    assert!(args.no_uv_unwrap);
}

/// Without the flags the textured-GLB export keeps the upstream defaults:
/// stage enabled (until `--no-texture-export`), bake 2048, unwrap on.
#[test]
fn run_trellis2_texture_export_flags_default_to_upstream() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "trellis2".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--output".to_string(),
        "out.glb".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(None, args.texture_size);
    assert!(!args.no_texture_export);
    assert!(!args.no_uv_unwrap);
}

/// Out-of-range `--texture-size` values fail at dispatch BEFORE loading any
/// weights (option assembly validates the 64..=8192 CLI range).
#[test]
fn run_model_rejects_invalid_trellis2_texture_size_at_dispatch() {
    for size in ["0", "32", "63", "8193"] {
        let cli = Cli::parse_from(vec![
            "lux3d".to_string(),
            "run".to_string(),
            "trellis2".to_string(),
            "--device".to_string(),
            "cpu".to_string(),
            "--source".to_string(),
            "does-not-exist.png".to_string(),
            "--texture-size".to_string(),
            size.to_string(),
            "--output".to_string(),
            "out.glb".to_string(),
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = run_model(args).expect_err(&format!(
            "`--texture-size {size}` must be rejected at dispatch"
        ));
        let message = error.to_string();
        assert!(
            message.contains("`--texture-size` must be in 64..=8192"),
            "unexpected rejection message for {size}: {message}"
        );
    }
}

/// The TRELLIS.2-only textured-GLB flags must be rejected for every other
/// family at dispatch BEFORE loading any weights (same fail-fast pattern as
/// the TripoSG-only and VGGT-only flags).
#[test]
fn run_model_rejects_trellis2_texture_flags_for_other_families() {
    for (family, extra) in [
        ("pi3", vec!["--texture-size", "1024"]),
        ("pi3x", vec!["--no-texture-export"]),
        ("triposr", vec!["--no-uv-unwrap"]),
        ("triposg", vec!["--texture-size", "4096"]),
        ("vggt", vec!["--no-texture-export"]),
    ] {
        let mut argv = vec![
            "lux3d".to_string(),
            "run".to_string(),
            family.to_string(),
            "--device".to_string(),
            "cpu".to_string(),
            "--source".to_string(),
            "does-not-exist".to_string(),
            "--output".to_string(),
            "out.ply".to_string(),
        ];
        argv.extend(extra.iter().map(|value| value.to_string()));
        let cli = Cli::parse_from(argv);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = run_model(args).expect_err(&format!(
            "TRELLIS.2-only flag must be rejected for {family}"
        ));
        let message = error.to_string();
        assert!(
            message.contains("is not supported for") && message.contains(family),
            "unexpected rejection message for {family}: {message}"
        );
    }
}

// ---------------------------------------------------------------------------
// VGGT-only run flags (`VggtRunArgs`)
// ---------------------------------------------------------------------------

/// The VGGT-only flags parse from the CLI surface with their values intact.
#[test]
fn run_vggt_parses_vggt_flags() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "vggt".to_string(),
        "--source".to_string(),
        "frames".to_string(),
        "--preprocess-mode".to_string(),
        "pad".to_string(),
        "--vggt-dtype".to_string(),
        "bf16".to_string(),
        "--batch-size".to_string(),
        "8".to_string(),
        "--colmap-dir".to_string(),
        "sparse".to_string(),
        "--colmap-conf-threshold".to_string(),
        "3.25".to_string(),
        "--colmap-max-points".to_string(),
        "1234".to_string(),
        "--depth-points".to_string(),
        "depth.ply".to_string(),
        "--depth-conf-threshold".to_string(),
        "0.5".to_string(),
        "--track-queries".to_string(),
        r#"[{"frame":0,"x":10.5,"y":20.0}]"#.to_string(),
        "--track-output".to_string(),
        "tracks.json".to_string(),
        "--track-iters".to_string(),
        "6".to_string(),
        "--track-points-ply".to_string(),
        "queries.ply".to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);

    assert_eq!("vggt", cli.command.family_name());
    assert_eq!(Some(ModelFamily::Vggt), cli.command.family());

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(VggtPreprocessKind::Pad, args.vggt.preprocess_mode);
    assert_eq!(Some(DTypeKind::Bf16), args.vggt.vggt_dtype);
    assert_eq!(8, args.vggt.batch_size);
    assert_eq!(
        Some(std::path::PathBuf::from("sparse")),
        args.vggt.colmap_dir
    );
    assert_eq!(3.25, args.vggt.colmap_conf_threshold);
    assert_eq!(1234, args.vggt.colmap_max_points);
    assert_eq!(
        Some(std::path::PathBuf::from("depth.ply")),
        args.vggt.depth_points
    );
    assert_eq!(0.5, args.vggt.depth_conf_threshold);
    assert_eq!(
        Some(r#"[{"frame":0,"x":10.5,"y":20.0}]"#.to_string()),
        args.vggt.track_queries
    );
    assert_eq!(
        Some(std::path::PathBuf::from("tracks.json")),
        args.vggt.track_output
    );
    assert_eq!(6, args.vggt.track_iters);
    assert_eq!(
        Some(std::path::PathBuf::from("queries.ply")),
        args.vggt.track_points_ply
    );
}

/// Defaults keep the historical behavior: crop mode, f32 (unset dtype), a
/// single forward pass (batch-size 1), upstream COLMAP defaults and no
/// optional exports.
#[test]
fn run_vggt_flags_default_to_historical_behavior() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "vggt".to_string(),
        "--source".to_string(),
        "frames".to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(VggtPreprocessKind::Crop, args.vggt.preprocess_mode);
    assert_eq!(None, args.vggt.vggt_dtype);
    assert_eq!(1, args.vggt.batch_size);
    assert_eq!(None, args.vggt.colmap_dir);
    assert_eq!(5.0, args.vggt.colmap_conf_threshold);
    assert_eq!(100_000, args.vggt.colmap_max_points);
    assert_eq!(None, args.vggt.depth_points);
    assert_eq!(1.0, args.vggt.depth_conf_threshold);
    assert_eq!(None, args.vggt.track_queries);
    assert_eq!(None, args.vggt.track_output);
    assert_eq!(4, args.vggt.track_iters);
    assert_eq!(None, args.vggt.track_points_ply);
}

/// Unknown enum values fail clap parsing with the possible values listed.
#[test]
fn run_vggt_enum_flags_reject_unknown_values() {
    for (flag, value) in [("--preprocess-mode", "blur"), ("--vggt-dtype", "q8_0")] {
        let parsed = Cli::try_parse_from(vec![
            "lux3d".to_string(),
            "run".to_string(),
            "vggt".to_string(),
            "--source".to_string(),
            "frames".to_string(),
            flag.to_string(),
            value.to_string(),
            "--output".to_string(),
            "out.ply".to_string(),
        ]);
        let error = parsed.expect_err(&format!("{flag} {value} must be rejected"));
        let message = error.to_string();
        assert!(
            message.contains("invalid value") && message.contains("possible values"),
            "unexpected message for {flag} {value}: {message}"
        );
    }
}

/// Hyphen-led JSON and negative coordinates inside `--track-queries` parse at
/// the clap level (range validation happens at dispatch); a hyphen-leading
/// numeric value for the confidence thresholds is not accepted as a flag value.
#[test]
fn run_vggt_hyphen_values_parse_for_track_queries_but_not_thresholds() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "vggt".to_string(),
        "--source".to_string(),
        "frames".to_string(),
        "--track-queries".to_string(),
        r#"[{"frame":0,"x":-12.5,"y":3.0}]"#.to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);
    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(
        Some(r#"[{"frame":0,"x":-12.5,"y":3.0}]"#.to_string()),
        args.vggt.track_queries
    );

    let parsed = Cli::try_parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "vggt".to_string(),
        "--source".to_string(),
        "frames".to_string(),
        "--colmap-conf-threshold".to_string(),
        "-1.0".to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);
    assert!(
        parsed.is_err(),
        "a hyphen-leading threshold value must be rejected by clap"
    );
}

/// `--batch-size 0` fails fast at dispatch, before any weights are loaded.
#[test]
fn run_model_rejects_vggt_batch_size_zero_at_dispatch_before_weights() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "vggt".to_string(),
        "--device".to_string(),
        "cpu".to_string(),
        "--source".to_string(),
        "does-not-exist".to_string(),
        "--batch-size".to_string(),
        "0".to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    let error = run_model(args).expect_err("`--batch-size 0` must be rejected");
    assert!(
        error
            .to_string()
            .contains("`--batch-size` must be at least 1"),
        "unexpected rejection message: {error}"
    );
}

/// A VGGT-only flag on another family must be rejected at dispatch BEFORE
/// loading any weights (same fail-fast pattern as the TripoSG-only flags).
#[test]
fn run_model_rejects_vggt_only_flag_at_dispatch_before_weights() {
    for (family, extra) in [
        ("pi3", vec!["--colmap-dir", "sparse"]),
        (
            "pi3x",
            vec!["--track-queries", r#"[{"frame":0,"x":1.0,"y":2.0}]"#],
        ),
        ("triposr", vec!["--depth-points", "depth.ply"]),
        ("triposg", vec!["--vggt-dtype", "f16"]),
        ("trellis2", vec!["--preprocess-mode", "pad"]),
    ] {
        let mut argv = vec![
            "lux3d".to_string(),
            "run".to_string(),
            family.to_string(),
            "--device".to_string(),
            "cpu".to_string(),
            "--source".to_string(),
            "does-not-exist".to_string(),
            "--output".to_string(),
            "out.ply".to_string(),
        ];
        argv.extend(extra.iter().map(|value| value.to_string()));
        let cli = Cli::parse_from(argv);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error =
            run_model(args).expect_err(&format!("VGGT-only flag must be rejected for {family}"));
        let message = error.to_string();
        assert!(
            message.contains("is not supported for") && message.contains(family),
            "unexpected rejection message for {family}: {message}"
        );
    }
}
