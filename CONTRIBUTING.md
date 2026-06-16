# Contributing to RasterDB

Thank you for your interest in improving RasterDB. Contributions are welcome in the form of bug reports, documentation fixes, SQL examples, tests, performance work, and focused operator or planner improvements.

## How To Contribute

1. Fork the repository and create a branch for your change.
2. Keep changes focused and consistent with the existing code style.
3. Add or update SQL tests, scripts, or expected outputs when behavior changes.
4. Build locally before opening a pull request.
5. Describe the query shape, implementation, and validation steps in the pull request.

## Development Checklist

RasterDB depends on a system install of RasterDF:

```bash
cd ../rasterdf
sudo ./install.sh --system
cd ../rasterdb
./build.sh --release --log-level=info
```

For GPU execution changes, include the Vulkan device used, driver details when relevant, and whether DuckDB fallback was enabled.

## License

By contributing, you agree that your contribution is provided under the MIT License used by this project.
