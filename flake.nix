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
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python313
              pkgs.uv
              pkgs.just
              pkgs.ffmpeg
            ];

            env.LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.ffmpeg
              pkgs.stdenv.cc.cc.lib
            ] + ":/run/opengl-driver/lib";

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
