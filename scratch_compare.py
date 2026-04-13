import json

for fname, label in [('benchmark_20260413_102755.json', 'HIGH(94.2%)'), ('benchmark_20260413_112541.json', 'LOW(73.7%)')]:
    with open(f'benchmarks/results/{fname}', encoding='utf-8') as f:
        data = json.load(f)
    for r in data['reports']:
        print(f"\n=== {label} - {r['pdf_file']} ===")
        print("Field accuracies:", json.dumps(r['field_accuracy'], indent=2))
        print("Metrics breakdown:")
        for k,v in r['metrics_breakdown'].items():
            print(f"  {k}: acc={v['acc']}%  miss={v['mis']}%  wrong={v['wrg']}%  misplaced={v['mpl']}%  halluc={v['hal']}%")
        
        # Show per-beam details for mismatches
        print("\nPer-beam scores:")
        for i, b in enumerate(r['beam_details']):
            score = b['score']
            status = b['status']
            exp_id = (b.get('expected') or {}).get('beam_id', '?')
            pred_id = (b.get('predicted') or {}).get('beam_id', '?')
            if score < 100:
                # Show which fields were wrong
                wrong_fields = []
                for f_name, f_score in b.get('field_scores', {}).items():
                    if f_score < 1.0:
                        wrong_fields.append(f"{f_name}={f_score}")
                print(f"  #{i+1} [{status}] exp={exp_id} pred={pred_id} score={score}%  wrong: {', '.join(wrong_fields)}")
