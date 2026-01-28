{
  description = "can-transcribe: Audio/video transcription CLI";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          opencv = pkgs.python313Packages.opencv4;

          # Base packages shared across all shells
          basePackages = [
            pkgs.python313
            pkgs.uv
            pkgs.just
            pkgs.ffmpeg-full
            opencv
          ];

          # Base libraries for LD_LIBRARY_PATH
          baseLibs = [
            pkgs.ffmpeg-full
            pkgs.stdenv.cc.cc.lib
            pkgs.dav1d
            opencv
          ];

          # Shell factory
          mkDevShell =
            {
              extraPackages ? [ ],
              extraLibs ? [ ],
              extraEnv ? { },
              uvExtra,
            }:
            pkgs.mkShell {
              packages = basePackages ++ extraPackages;

              env = {
                LD_LIBRARY_PATH = lib.makeLibraryPath (baseLibs ++ extraLibs);
              } // extraEnv;

              shellHook = ''
                export PYTHONPATH="${opencv}/${pkgs.python313.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}"
                uv sync --extra ${uvExtra}
                . .venv/bin/activate
              '';
            };
        in
        {
          # NVIDIA CUDA (x86_64-linux only)
          nvidia = mkDevShell {
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
            uvExtra = "nvidia";
          };

          # AMD ROCm (x86_64-linux only)
          amd = mkDevShell {
            extraPackages = [
              pkgs.rocmPackages.clr
              pkgs.rocmPackages.rocm-smi
            ];
            extraLibs = [
              pkgs.rocmPackages.clr
            ];
            extraEnv = {
              HSA_OVERRIDE_GFX_VERSION = "11.0.0"; # Adjust for your GPU
            };
            uvExtra = "amd";
          };

          # Apple Silicon (aarch64-darwin)
          apple = mkDevShell {
            extraPackages = [ ];
            extraLibs = [ ];
            extraEnv = { };
            uvExtra = "apple";
          };

          # CPU only (no GPU acceleration, smaller download)
          cpu = mkDevShell {
            extraPackages = [ ];
            extraLibs = [ ];
            extraEnv = { };
            uvExtra = "cpu";
          };

          # Default: CPU shell
          default = mkDevShell {
            extraPackages = [ ];
            extraLibs = [ ];
            extraEnv = { };
            uvExtra = "cpu";
          };
        }
      );
    };
}
