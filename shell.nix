let
  pkgs = import <nixpkgs> { };

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

  butterflowPackage = python.pkgs.buildPythonPackage rec {
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

in pkgs.mkShell { 
  packages = [ 
    pythonEnv
    butterflowPackage
    pkgs.uv
    python.pkgs.pytest
    ]; 
}

