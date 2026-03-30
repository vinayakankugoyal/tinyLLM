{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShellNoCC {
    packages = [
        (pkgs.python3.withPackages (ps: [
            ps.jax
            ps.optax
        ]))
        pkgs.ruff
    ];
    shellHook = ''
        echo "Welcome to tinyLLM!"
        python3 tinyLLM.py --help
    '';
}