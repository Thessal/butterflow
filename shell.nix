let
  pkgs = import <nixpkgs> {
    overlays = [
      (import (fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))
    ];
  };

  rustVersion = "1.91.1";
  myRust = pkgs.rust-bin.stable.${rustVersion}.default.override {
    extensions = [
      "rust-src" # for rust-analyzer
      "rust-analyzer"
    ];
  };

  inherit (pkgs) lib;

  pyproject-nix = import (builtins.fetchGit {
    url = "https://github.com/pyproject-nix/pyproject.nix.git";
  }) {
    inherit lib;
  };
  
  project = pyproject-nix.lib.project.loadPyproject {
    projectRoot = ./.;
  };

  python = pkgs.python313;
  arg = project.renderers.withPackages { inherit python; };
  pythonEnv = python.withPackages arg;

  butterflowOld = python.pkgs.buildPythonPackage rec {
    pname = "butterflow"; 
    version = "0.1.0"; 
    src = ./.; 

    buildInputs = with python.pkgs; [
      hatchling
      pythonEnv
    ];
    
    format = "pyproject"; 
    doCheck = false; 
    doInstallCheck = false;
    dontCheck = true;
  };


  butterflow = pkgs.rustPlatform.buildRustPackage {
    pname = "butterflow";
    version = "0.1.0";
    src = ./.;
    cargoLock = {
      lockFile = ./Cargo.lock;
    };
  };

in pkgs.mkShell { 
  packages = [ 
    pkgs.antigravity 

    pythonEnv
    butterflowOld
    butterflow 
    pkgs.uv
    python.pkgs.pytest
    
    # Rust
    myRust
    pkgs.cargo
    pkgs.rustc
    pkgs.rustfmt
    pkgs.rust-analyzer
    pkgs.clippy
    ]; 
}

