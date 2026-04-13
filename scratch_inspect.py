import json

with open('benchmarks/results/benchmark_20260413_112541.json', encoding='utf-8') as f:
    data = json.load(f)
for r in data['reports']:
    for b in r['beam_details']:
        if b['status'] == 'MATCHED' and b['score'] < 85:
            exp = b.get('expected', {})
            pred = b.get('predicted', {})
            eid = exp.get('beam_id', '?')
            print(f"=== {eid} (score={b['score']}%) ===")
            for key in ['bottom_main_bars_left','bottom_main_bars_mid','bottom_main_bars_right',
                        'lap_length_top_left','lap_length_top_right','lap_length_bottom_left','lap_length_bottom_right',
                        'stirrups_left','stirrups_middle','stirrups_right','face_bars','dimensions']:
                e = exp.get(key, '')
                p = pred.get(key, '')
                if str(e) != str(p):
                    print(f"  {key}: expected={e}  predicted={p}")
            print()
