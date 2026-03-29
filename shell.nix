{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShellNoCC {
    packages = [
        (pkgs.python3.withPackages (ps: [
            ps.jax
            ps.optax
        ]))
    ];
    shellHook = ''
        echo "Welcome to tiny!"
        python3 tiny.py --help
    '';
}