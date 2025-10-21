.PHONY: test example

test:
	pytest -q

example:
	python3 core/topology/topology_generator.py grid -n 5 -s 10 -o sample_topologies/grid5.json
	python3 core/csc/csc_generator.py sample_topologies/grid5.json artifacts/example/sim.csc --build-root artifacts/example/build
	@echo "Generated artifacts/example/sim.csc"


