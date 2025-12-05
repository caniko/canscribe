{
  description = "Flake for developing Python on Nix with uv";

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
            config.allowUnfree = true;  # Required for CUDA toolkit
          };
          cudaToolkit = pkgs.cudaPackages.cudatoolkit;
          opencv = pkgs.python313Packages.opencv4Full;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python313
              pkgs.uv
              pkgs.just
              pkgs.ffmpeg-full  # Full ffmpeg with all codecs including AV1 (dav1d) and hardware acceleration
              opencv  # OpenCV with VAAPI/CUDA support
              # For building flash-attn
              cudaToolkit
              pkgs.ninja
              pkgs.gcc
            ];

            env = {
              LD_LIBRARY_PATH = lib.makeLibraryPath [
                pkgs.ffmpeg-full
                pkgs.stdenv.cc.cc.lib
                pkgs.dav1d  # AV1 software decoder
                cudaToolkit
                opencv
              ] + ":/run/opengl-driver/lib";
              CUDA_HOME = "${cudaToolkit}";
            };

            shellHook = ''
              # Add Nix OpenCV to Python path (before venv activation)
              export PYTHONPATH="${opencv}/${pkgs.python313.sitePackages}:$PYTHONPATH"
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
