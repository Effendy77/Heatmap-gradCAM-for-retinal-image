.PHONY: verify select gradcam panels provenance all

verify:
	python scripts/verify_inputs.py --config configs/hybrid.yaml

select:
	python scripts/select_cases.py --config configs/hybrid.yaml --out selections/

gradcam:
	python scripts/generate_gradcam.py --config configs/hybrid.yaml --selections selections/ --out outputs/gradcam/

panels:
	python scripts/compose_panels.py --root outputs/gradcam/ --out panels/

provenance:
	python scripts/provenance_log.py --config configs/hybrid.yaml --panels_dir panels/ --out run_provenance.json

all: verify select gradcam panels provenance
