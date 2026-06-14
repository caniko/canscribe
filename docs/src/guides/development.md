# Development

Enter the development shell:

```sh
nix develop
```

Run the main checks:

```sh
uv run mypy .
uv run ruff format .
uv run pytest
```

The Nix shell also exposes helper commands:

```sh
canscribe-sync
canscribe-test
canscribe-test-live
canscribe-typecheck
canscribe-smoke
```

## Documentation and Site

Build the mdBook documentation:

```sh
nix build .#docs
```

Build the combined Codeberg Pages site:

```sh
nix build .#site
```

The combined site puts the Plinth landing page at the deployment root and the mdBook documentation under `/docs/`.
