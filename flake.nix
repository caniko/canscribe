{
  description = "canscribe: Audio/video transcription CLI";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    py-harbor = {
      url = "git+https://codeberg.org/caniko/py-harbor.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    plinth = {
      url = "git+https://codeberg.org/caniko/plinth";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix.url = "github:numtide/treefmt-nix";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      py-harbor,
      plinth,
      treefmt-nix,
      git-hooks,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      py = py-harbor.lib;

      mkRuntime =
        pkgs:
        let
          ffmpeg = py.mkFfmpegCompat { inherit pkgs; };
          opencv = pkgs.python313Packages.opencv4;
        in
        {
          inherit ffmpeg opencv;

          basePackages = [
            pkgs.just
            ffmpeg
            opencv
          ];

          baseLibs = [
            ffmpeg
            pkgs.stdenv.cc.cc.lib
            pkgs.dav1d
            pkgs.libdrm
            pkgs.libsndfile
            pkgs.zlib # triton's _C extension
            pkgs.zstd
            opencv
          ]
          ++ lib.optionals pkgs.stdenv.isLinux [
            pkgs.libGL
            pkgs.libxkbcommon
            pkgs.xorg.libX11
            pkgs.xorg.libXext
            pkgs.xorg.libSM
            pkgs.xorg.libICE
            pkgs.xorg.libxcb
          ];
        };

      mkSmokeCommand =
        pkgs: runtime:
        pkgs.writeShellApplication {
          name = "canscribe-smoke";
          runtimeInputs = [
            pkgs.uv
            runtime.ffmpeg
          ];
          text = ''
            uv run --no-sync python - <<'PY'
            import importlib.metadata as md
            import os
            import shutil
            import subprocess
            import sys
            import torch

            for package in ("torch", "torchaudio", "torchvision", "torchcodec"):
                try:
                    print(f"{package}: {md.version(package)}")
                except md.PackageNotFoundError:
                    print(f"{package}: MISSING")

            ffmpeg = shutil.which("ffmpeg")
            print(f"ffmpeg: {ffmpeg or 'MISSING'}")
            if ffmpeg:
                print(subprocess.run([ffmpeg, "-version"], check=False, capture_output=True, text=True).stdout.splitlines()[0])

            gpu = torch.cuda.is_available()
            backend = "none"
            if gpu and getattr(torch.version, "hip", None):
                backend = "rocm"
            elif gpu and torch.version.cuda:
                backend = "cuda"
            elif gpu:
                backend = "unknown-gpu"
            elif torch.backends.mps.is_available():
                backend = "mps"
            print(f"torch gpu available: {gpu}")
            print(f"backend: {backend}")
            if gpu:
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    arch = getattr(props, "gcnArchName", "unknown")
                    gib = props.total_memory / 1024 ** 3
                    print(f"device {index}: {torch.cuda.get_device_name(index)}; arch={arch}; memory={gib:.1f} GiB")
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import torch; x=torch.ones((1,), device='cuda'); y=x + 1; torch.cuda.synchronize(); print(float(y.cpu()[0]))",
                    ],
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if probe.returncode == 0:
                    print(f"gpu kernel probe: OK ({probe.stdout.strip()})")
                else:
                    stderr = probe.stderr.strip().splitlines()
                    detail = f"returncode {probe.returncode}"
                    if stderr:
                        detail = f"{detail}; stderr: {stderr[-1]}"
                    print(f"gpu kernel probe: FAIL ({detail})")
                    raise SystemExit(1)
            PY
          '';
        };

      mkPyprojectOverrides =
        {
          pkgs,
          python,
          runtime,
        }:
        final: prev: {
          insightface = prev.insightface.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              final.setuptools
              final.numpy
              final.cython
            ];
          });
          numba = prev.numba.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.tbb
            ];
          });
          julius = py.pythonOverrides.addSetuptools final prev "julius";
          torchaudio = py.pythonOverrides.addTorchRuntime python final prev "torchaudio";
          torchcodec = py.pythonOverrides.addTorchCodecFfmpegRuntime {
            inherit python;
            ffmpeg = runtime.ffmpeg;
          } final prev;
          torchvision = py.pythonOverrides.addTorchRuntime python final prev "torchvision";
        };

      mkDevShells =
        system:
        let
          pkgs = py.mkPkgs { inherit system; };
          python = pkgs.python313;
          runtime = mkRuntime pkgs;
          isX86Linux = system == "x86_64-linux";
          isAarch64Darwin = system == "aarch64-darwin";
          smokeCommand = mkSmokeCommand pkgs runtime;
          treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
          pre-commit-check = git-hooks.lib.${system}.run {
            src = ./.;
            hooks = import ./nix/pre-commit.nix {
              inherit pkgs;
              treefmtWrapper = treefmtEval.config.build.wrapper;
            };
          };

          # Diagnostic tools only. The ROCm PyTorch wheel bundles its own ROCm
          # userspace; exposing nixpkgs ROCm libs via LD_LIBRARY_PATH (or clr's
          # setup hook exporting HIP_DEVICE_LIB_PATH) shadows the wheel's
          # runtime with an incompatible version and segfaults on first kernel
          # launch. Keep clr/rocm-runtime out of the shell.
          rocmTools = with pkgs.rocmPackages; [
            rocminfo
            rocm-smi
          ];

          mkDevShell =
            {
              uvExtra,
              extraPackages ? [ ],
              extraLibs ? [ ],
              extraEnv ? { },
            }:
            let
              uvFlags = "--extra ${uvExtra} --group dev";
              helperPackages = [
                (py.mkUvHelper {
                  inherit pkgs;
                  name = "canscribe-sync";
                  command = "sync ${uvFlags}";
                })
                (py.mkUvHelper {
                  inherit pkgs;
                  name = "canscribe-test";
                  command = "run ${uvFlags} pytest";
                })
                (py.mkUvHelper {
                  inherit pkgs;
                  name = "canscribe-test-live";
                  command = "run ${uvFlags} pytest -m 'integration and not slow'";
                })
                (py.mkUvHelper {
                  inherit pkgs;
                  name = "canscribe-typecheck";
                  command = "run ${uvFlags} mypy .";
                })
                smokeCommand
              ];
            in
            py.mkUvDevShell {
              inherit
                pkgs
                python
                uvExtra
                helperPackages
                extraEnv
                ;
              basePackages = runtime.basePackages ++ [ pkgs.mdbook ];
              extraPackages = pre-commit-check.enabledPackages ++ extraPackages;
              baseLibs = runtime.baseLibs;
              extraLibs = extraLibs;
              pythonPathEntries = [ "${runtime.opencv}/${python.sitePackages}" ];
              shellHookSuffix = pre-commit-check.shellHook;
            };

          cpuShell = mkDevShell {
            uvExtra = "cpu";
          };

          appleShell = mkDevShell {
            uvExtra = "apple";
          };
        in
        {
          docs = pkgs.mkShell {
            packages = [ pkgs.mdbook ];
          };

          site = pkgs.mkShell {
            packages = [
              pkgs.mdbook
              plinth.packages.${system}.plinth-project
            ];
          };

          cpu = cpuShell;
          default = if isAarch64Darwin then appleShell else cpuShell;
        }
        // lib.optionalAttrs isAarch64Darwin {
          apple = appleShell;
        }
        // lib.optionalAttrs isX86Linux {
          amd = mkDevShell {
            uvExtra = "amd";
            extraPackages = rocmTools;
            extraEnv = {
              ROCR_VISIBLE_DEVICES = "0";
              HIP_VISIBLE_DEVICES = "0";
              CUDA_VISIBLE_DEVICES = "0";
            };
          };

          nvidia = mkDevShell {
            uvExtra = "nvidia";
            extraPackages = [
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cudnn
              pkgs.ninja
              pkgs.gcc
            ];
            extraLibs = [
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cudnn
              pkgs.addDriverRunpath.driverLink
            ];
            extraEnv = {
              CUDA_HOME = "${pkgs.cudaPackages.cuda_nvcc}";
            };
          };
        };

      mkPythonPackage =
        system:
        let
          pkgs = py.mkPkgs { inherit system; };
          python = pkgs.python313;
          runtime = mkRuntime pkgs;
          cpuDependencySpec = {
            canscribe = [ "cpu" ];
          };
        in
        py.mkUvAppPackage {
          inherit pkgs python;
          name = "canscribe-cpu";
          envName = "canscribe-cpu-env";
          workspaceRoot = ./.;
          dependencies = cpuDependencySpec;
          scripts = [
            "canscribe"
            "ct"
          ];
          runtimePathPackages = [ runtime.ffmpeg ];
          runtimeLibraryPackages = runtime.baseLibs;
          pyprojectOverrides = mkPyprojectOverrides { inherit pkgs python runtime; };
        };

      mkPythonCheckEnv =
        system:
        let
          pkgs = py.mkPkgs { inherit system; };
          python = pkgs.python313;
          runtime = mkRuntime pkgs;
          checkDependencySpec = {
            canscribe = [
              "cpu"
              "dev"
            ];
          };
        in
        py.mkUvCheckEnv {
          inherit pkgs python;
          name = "canscribe-cpu-check-env";
          workspaceRoot = ./.;
          dependencies = checkDependencySpec;
          pyprojectOverrides = mkPyprojectOverrides { inherit pkgs python runtime; };
        };

      mkChecks =
        system:
        let
          pkgs = py.mkPkgs { inherit system; };
          runtime = mkRuntime pkgs;
          canscribe = self.packages.${system}.canscribe-cpu;
          checkEnv = mkPythonCheckEnv system;
          treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
          ffmpegAbiCheck = py.mkFfmpegTorchCodecAbiCheck {
            inherit pkgs;
            ffmpeg = runtime.ffmpeg;
            name = "canscribe-ffmpeg-torchcodec-abi-check";
          };
        in
        {
          flake-eval = pkgs.runCommand "canscribe-flake-eval" { } ''
            test -x ${canscribe}/bin/canscribe
            test -x ${runtime.ffmpeg}/bin/ffmpeg
            test -e ${ffmpegAbiCheck}/result
            mkdir -p $out
            echo ok > $out/result
          '';

          formatting = treefmtEval.config.build.check self;

          offline-tests = pkgs.runCommand "canscribe-offline-tests" { } ''
            export HOME=$TMPDIR/home
            export XDG_CACHE_HOME=$TMPDIR/cache
            export LD_LIBRARY_PATH=${lib.makeLibraryPath runtime.baseLibs}
            mkdir -p "$HOME" "$XDG_CACHE_HOME" "$out"
            cd ${./.}
            ${checkEnv}/bin/python -m pytest -p no:cacheprovider
            echo ok > $out/result
          '';

          typecheck = pkgs.runCommand "canscribe-typecheck" { } ''
            export HOME=$TMPDIR/home
            export XDG_CACHE_HOME=$TMPDIR/cache
            export LD_LIBRARY_PATH=${lib.makeLibraryPath runtime.baseLibs}
            mkdir -p "$HOME" "$XDG_CACHE_HOME" "$out"
            cd ${./.}
            ${checkEnv}/bin/python -m mypy .
            echo ok > $out/result
          '';
        };

      mkDocs =
        system:
        let
          pkgs = py.mkPkgs { inherit system; };
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "canscribe-docs";
          version = "0.1.0";
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.maybeMissing ./docs;
          };
          nativeBuildInputs = [ pkgs.mdbook ];
          phases = [
            "buildPhase"
            "installPhase"
          ];
          buildPhase = ''
            cp -r --no-preserve=mode $src/docs docs
            mdbook build docs
          '';
          installPhase = ''
            cp -r docs/book $out
          '';
        };

      mkSite =
        system:
        let
          pkgs = py.mkPkgs { inherit system; };
          docs = self.packages.${system}.docs;
          plinthProject = plinth.packages.${system}.plinth-project;
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "canscribe-site";
          version = "0.1.0";
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              (lib.fileset.maybeMissing ./website)
            ];
          };
          nativeBuildInputs = [ plinthProject ];
          phases = [
            "buildPhase"
            "installPhase"
          ];
          buildPhase = ''
            cp -r --no-preserve=mode $src/website website
            plinth-project build --config website/plinth-project.toml --out public
          '';
          installPhase = ''
            mkdir -p $out
            cp -r public/. $out/
            mkdir -p $out/docs
            cp -r ${docs}/. $out/docs/
          '';
        };
    in
    {
      devShells = py.forAllSystems mkDevShells;

      packages = py.forPackageSystems (
        system:
        let
          canscribeCpu = mkPythonPackage system;
        in
        {
          canscribe-cpu = canscribeCpu;
          docs = mkDocs system;
          site = mkSite system;
          default = canscribeCpu;
        }
      );

      apps = py.forPackageSystems (
        system:
        let
          canscribeCpu = self.packages.${system}.canscribe-cpu;
        in
        {
          canscribe-cpu = {
            type = "app";
            program = "${canscribeCpu}/bin/canscribe";
          };
          default = self.apps.${system}.canscribe-cpu;
        }
      );

      formatter = py.forAllSystems (
        system:
        let
          pkgs = py.mkPkgs { inherit system; };
          treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
        in
        treefmtEval.config.build.wrapper
      );

      checks = py.forPackageSystems mkChecks;
    };
}
