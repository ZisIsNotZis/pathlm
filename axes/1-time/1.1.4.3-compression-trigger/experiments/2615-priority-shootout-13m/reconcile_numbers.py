"""逐条核对 analysis.md / stability.md / README.md 里引用的数字与 results.json。

用法：在**本目录**里 `python3 reconcile_numbers.py`（纯 CPU，<1 s）。
每一项都是「报告里写的数」对「results.json 里的数」的断言，任何一处不一致就直接抛错——
用于兑现「报告里的每个成本/次数/等式都要与 results.json 逐一对得上」的纪律。
本脚本是 2615 契约 N 的交付物之一（分析数字的对账工具，不是实验代码）。
"""
import json, math
r = json.load(open('results.json'))
rows, cr, pb, cost = r['arms_table'], r['criteria'], r['pair_budget_diagnostic'], r['cost']
ok = 0
def chk(name, got, want, tol=1e-6):
    global ok
    assert got is not None, f"{name}: None"
    assert abs(got - want) <= tol, f"{name}: got {got} want {want}"
    ok += 1
def chk4(name, got, want, tol=None):
    chk(name, got, want, 5.1e-5 if tol is None else tol)   # analysis 表里按 4 位小数引用

# §0/§1 判据表
prim = {
 (8,'distance'):(0.196528,0.665039), (8,'lru:k=64'):(0.180614,0.682129), (8,'content'):(0.172165,0.664307),
 (32,'distance'):(0.640980,0.633545),(32,'lru:k=64'):(0.696677,0.647705),(32,'content'):(0.661762,0.616211),
 (64,'distance'):(1.248019,0.594971),(64,'lru:k=64'):(1.330958,0.583008),(64,'content'):(1.210212,0.578857)}
ids = {(8,'distance'):'distance:N=8',(8,'lru:k=64'):'lru:k=64:N=8',(8,'content'):'content:N=8',
       (32,'distance'):'distance:N=32',(32,'lru:k=64'):'lru:k=64:N=32',(32,'content'):'content:N=32',
       (64,'distance'):'distance:N=64',(64,'lru:k=64'):'lru:k=64:N=64',(64,'content'):'content:N=64'}
for k,(dmg,rec) in prim.items():
    chk(f"§1 {ids[k]} damage", rows[ids[k]]['damage_bpc'], dmg)
    chk(f"§1 {ids[k]} recall", rows[ids[k]]['retention']['recall'], rec)
for N,val in [(8,{'lru':0.015914,'content':0.024363})]:
    for rule,v in val.items():
        chk(f"§1 gain N={N} {rule}", cr['per_N'][0]['damage_gain_vs_distance'][rule], v)
chk("§1 gain N=32 lru", cr['per_N'][1]['damage_gain_vs_distance']['lru'], -0.055697)
chk("§1 gain N=32 content", cr['per_N'][1]['damage_gain_vs_distance']['content'], -0.020782)
chk("§1 gain N=64 lru", cr['per_N'][2]['damage_gain_vs_distance']['lru'], -0.082939)
chk("§1 gain N=64 content", cr['per_N'][2]['damage_gain_vs_distance']['content'], 0.037807)
chk("§1 recall gain N=8 lru", cr['per_N'][0]['recall_gain_vs_distance']['lru'], 0.01709)
chk("§1 recall gain N=8 content", cr['per_N'][0]['recall_gain_vs_distance']['content'], -0.000732)
chk("§1 recall gain N=32 lru", cr['per_N'][1]['recall_gain_vs_distance']['lru'], 0.01416)
chk("§1 recall gain N=32 content", cr['per_N'][1]['recall_gain_vs_distance']['content'], -0.017334)
chk("§1 recall gain N=64 lru", cr['per_N'][2]['recall_gain_vs_distance']['lru'], -0.011963)
chk("§1 recall gain N=64 content", cr['per_N'][2]['recall_gain_vs_distance']['content'], -0.016114)
for i,N in enumerate((8,32,64)):
    assert cr['per_N'][i]['recall_criterion_arithmetically_reachable'] is False; ok+=1
    for rule in ('distance','lru','content'):
        chk(f"§1 budget N={N} {rule}", cr['per_N'][i]['same_budget_n_dropped_keys'][rule], float(N))
chk("§1 baseline recall", cr['baseline_recall_full'], 0.686768)
chk("§1 baseline bpc", cr['baseline_bpc_full'], 1.483569)
assert cr['verdict']=='priority_rules_informative'; ok+=1
# §2 表
chk4("§2 qmin128 d8", rows['distance:N=8']['damage_qmin128'], 0.0085)
chk4("§2 qmin128 d32", rows['distance:N=32']['damage_qmin128'], 0.0256)
chk4("§2 qmin128 d64", rows['distance:N=64']['damage_qmin128'], 0.0668)
chk4("§2 qmin128 l8", rows['lru:k=64:N=8']['damage_qmin128'], 0.0058)
chk4("§2 qmin128 l32", rows['lru:k=64:N=32']['damage_qmin128'], 0.0328)
chk4("§2 qmin128 l64", rows['lru:k=64:N=64']['damage_qmin128'], 0.1300)
chk4("§2 qmin128 c8", rows['content:N=8']['damage_qmin128'], 0.0070)
chk4("§2 qmin128 c32", rows['content:N=32']['damage_qmin128'], 0.0329)
chk4("§2 qmin128 c64", rows['content:N=64']['damage_qmin128'], 0.0672)
chk4("§2 qmin200 d8", rows['distance:N=8']['damage_qmin200'], 0.0082)
chk4("§2 qmin200 d32", rows['distance:N=32']['damage_qmin200'], 0.0211)
chk4("§2 qmin200 d64", rows['distance:N=64']['damage_qmin200'], 0.0442)
chk4("§2 qmin200 l8", rows['lru:k=64:N=8']['damage_qmin200'], 0.0013)
chk4("§2 qmin200 l32", rows['lru:k=64:N=32']['damage_qmin200'], 0.0130)
chk4("§2 qmin200 l64", rows['lru:k=64:N=64']['damage_qmin200'], 0.0415)
chk4("§2 qmin200 c8", rows['content:N=8']['damage_qmin200'], 0.0051)
chk4("§2 qmin200 c32", rows['content:N=32']['damage_qmin200'], 0.0260)
chk4("§2 qmin200 c64", rows['content:N=64']['damage_qmin200'], 0.0475)
for cid,dv in [('distance:N=8',0.0977),('distance:N=32',0.0836),('distance:N=64',0.0872),
               ('lru:k=64:N=8',0.1156),('lru:k=64:N=32',0.1051),('lru:k=64:N=64',0.1041),
               ('content:N=8',0.0926),('content:N=32',0.0908),('content:N=64',0.0864)]:
    chk(f"§2 per1k {cid}", 1000*rows[cid]['damage_bpc']/rows[cid]['activity']['n_dropped_pairs_mean'], dv, 1e-4)
    r_ = rows[cid]['retention']
    pass
# §2 配对/共同可见
for cid,dv in [('distance:N=8',0.458661),('distance:N=32',0.443979),('distance:N=64',0.425127),
               ('lru:k=64:N=8',0.590296),('lru:k=64:N=32',0.528),('lru:k=64:N=64',0.478678),
               ('content:N=8',0.513324),('content:N=32',0.456476),('content:N=64',0.433298)]:
    chk(f"§2 broken {cid}", rows[cid]['retention']['recall_plant_broken'], dv)
for cid,n in [('distance:N=8',508),('distance:N=32',955),('distance:N=64',1576),
              ('lru:k=64:N=8',371),('lru:k=64:N=32',1250),('lru:k=64:N=64',2087),
              ('content:N=8',713),('content:N=32',1413),('content:N=64',1874)]:
    chk(f"§2 nDel {cid}", rows[cid]['retention']['n_deleted_needles'], n)
for N,vals in [(8,(0.707832,0.699708,0.701007,3077)),(32,(0.719553,0.707539,0.710025,2414)),
               (64,(0.725287,0.701256,0.711087,1831))]:
    d = r['common_visible_subset'][f'N={N}']
    chk(f"§2 common n N={N}", d['n'], vals[3])
    for i,rule in enumerate((f'distance:N={N}', f'lru:k=64:N={N}', f'content:N={N}')):
        chk(f"§2 common {rule}", d['recall'][rule], vals[i])
# §3 复用
rc = r['reuse_check_contract_M']
assert rc['max_abs_diff'] < 1e-6 and rc['checkpoint_same']; ok+=1
for p,w in zip(rc['points'],[1.680097,2.124549,2.731588]):
    chk(f"§3 m {p['config']}", p['m_recorded_bpc'], w); chk(f"§3 this {p['config']}", p['this_run_bpc'], w)
chk("§3 shared vs per-sample max", rc['max_shared_vs_per_sample_diff'], 2.45e-07, 1e-8)
# §4 桶
chk("§4 base recall", r['baseline']['retention']['recall'], 0.686768)
chk4("§4 base window", r['baseline']['retention']['buckets']['window']['recall'], 0.694537)
chk4("§4 base anchor", r['baseline']['retention']['buckets']['anchor']['recall'], 0.6319)
chk("§4 floor", r['no_needle_control']['recall'], 0.433105)
chk("§4 d8 anchor", rows['distance:N=8']['retention']['buckets']['anchor']['recall'], 0.458661)
chk("§4 d8 window", rows['distance:N=8']['retention']['buckets']['window']['recall'], 0.694259)
chk("§4 l8 anchor", rows['lru:k=64:N=8']['retention']['buckets']['anchor']['recall'], 0.627953)
chk("§4 l8 window", rows['lru:k=64:N=8']['retention']['buckets']['window']['recall'], 0.69667)
chk("§4 l8 beyond", rows['lru:k=64:N=8']['retention']['buckets']['beyond']['recall'], 0.6)
chk("§4 c8 anchor", rows['content:N=8']['retention']['buckets']['anchor']['recall'], 0.545276)
chk("§4 c8 window", rows['content:N=8']['retention']['buckets']['window']['recall'], 0.699243)
chk("§4 c8 beyond", rows['content:N=8']['retention']['buckets']['beyond']['recall'], 0.543269)
chk("§4 paired base d8", rows['distance:N=8']['retention']['paired_baseline_recall_deleted'], 0.63189)
chk("§4 paired deleted d8", rows['distance:N=8']['retention']['paired_recall_deleted'], 0.458661)
for cid,k,v in [('lru:k=64:N=8','paired_baseline',0.617251),('content:N=8','paired_baseline',0.639551),
                ('distance:N=64','paired_baseline',0.645305),('lru:k=64:N=64','paired_baseline',0.648299),
                ('content:N=64','paired_baseline',0.646211),('distance:N=64','paired_recall',0.425127),
                ('lru:k=64:N=64','paired_recall',0.478678),('content:N=64','paired_recall',0.433298),
                ('lru:k=64:N=8','paired_drop',0.026955),('content:N=8','paired_drop',0.126227),
                ('distance:N=8','paired_drop',0.173229),('distance:N=64','paired_drop',0.220178),
                ('lru:k=64:N=64','paired_drop',0.169621),('content:N=64','paired_drop',0.212913)]:
    f = {'paired_baseline':'paired_baseline_recall_deleted','paired_recall':'paired_recall_deleted',
         'paired_drop':'paired_drop_deleted'}[k]
    chk4(f"§4 {cid} {k}", rows[cid]['retention'][f], v)
for cid,k,v in [('distance:N=64','anchor',0.415354),('distance:N=64','window',0.70119),
                ('distance:N=64','beyond',0.429775),('lru:k=64:N=64','anchor',0.48622),
                ('lru:k=64:N=64','window',0.690779),('lru:k=64:N=64','beyond',0.483077),
                ('content:N=64','anchor',0.425197),('content:N=64','window',0.70162),
                ('content:N=64','beyond',0.43631)]:
    chk4(f"§4 {cid} {k}", rows[cid]['retention']['buckets'][k]['recall'], v)
# §5 分布
aq = r['attention_quality']['mass_total']['by_bin']
for b,v in [('[0,8)',2.33895062),('[8,32)',1.03897433),('[32,64)',0.92310719),('[64,128)',0.85298245),('[128,192)',0.78381879),('[192,256)',0.5947179)]:
    chk(f"§5 mass {b}", aq[b], v, 1e-7)
mr = r['attention_quality']['mass_recent']['64']['by_bin']
for b,v in [('[0,8)',0.07193522),('[8,32)',0.02528144),('[32,64)',0.0290823),('[64,128)',0.04606521),('[128,192)',0.18219156),('[192,256)',0.5947179)]:
    chk(f"§5 recent64 {b}", mr[b], v, 1e-7)
cq = r['attention_quality']['cos_query']['by_bin']; kn = r['attention_quality']['key_norm']['by_bin']; sr = r['attention_quality']['soft_recency']['by_bin']
for b,v in [('[0,8)',-0.42802946),('[8,32)',-0.44678879),('[32,64)',-0.41588813),('[64,128)',-0.21670082),('[128,192)',0.20027583),('[192,256)',0.37169637)]:
    chk(f"§5 cos {b}", cq[b], v, 1e-7)
for b,v in [('[0,8)',17.89076594),('[8,32)',16.15077009),('[32,64)',12.67317108),('[64,128)',8.67914831),('[128,192)',10.13571636),('[192,256)',16.27058483)]:
    chk(f"§5 kn {b}", kn[b], v, 1e-7)
for b,v in [('[0,8)',34.11323635),('[8,32)',46.6865062),('[32,64)',73.04680049),('[64,128)',117.22877332),('[128,192)',175.23518543),('[192,256)',230.79608417)]:
    chk(f"§5 rec {b}", sr[b], v, 1e-7)
q = r['attention_quality']['mass_total']['quantiles']
for kk,v in [('min',0.0),('p25',0.7366813),('median',0.81719157),('p75',0.89329318),('max',9.37432734)]:
    chk(f"§5 quant {kk}", q[kk], v, 1e-7)
chk("§5 zero_frac", r['attention_quality']['mass_total']['zero_frac'], 0.003906)
print('§5 top10', [(h['pos'],h['mass']) for h in r['attention_quality']['mass_total']['top_positions']])
# §6 算术
chk("§6 N=8 ceiling need", 0.665039+0.10, 0.765039, 1e-6)
chk("§6 N=32 ceiling need", 0.633545+0.10, 0.733545, 1e-6)
chk("§6 N=64 ceiling need", 0.594971+0.10, 0.694971, 1e-6)
chk("§6 pairs N=64 closed form", sum(255-j for j in range(64)), 14304)
for cid,v in [('distance:N=8',2012.0),('lru:k=64:N=8',1561.846354),('content:N=8',1859.429688),
              ('lru:k=64:N=64',12785.960938),('content:N=64',14006.6875)]:
    chk(f"§6 pairs {cid}", rows[cid]['activity']['n_dropped_pairs_mean'], v, 1e-5)
# §6.4 对齐后
pm = {p['id']:p for p in pb['pair_matched']}
chk4("§6.4 lru N=8 matched", pm['lru:k=64:N=8']['gain_vs_distance_matched_pairs'], -0.0170, 1e-4)
chk4("§6.4 content N=8 matched", pm['content:N=8']['gain_vs_distance_matched_pairs'], 0.0132, 1e-4)
chk4("§6.4 lru N=32 matched", pm['lru:k=64:N=32']['gain_vs_distance_matched_pairs'], -0.1384, 1e-4)
chk4("§6.4 content N=32 matched", pm['content:N=32']['gain_vs_distance_matched_pairs'], -0.0507, 1e-4)
chk4("§6.4 lru N=64 matched", pm['lru:k=64:N=64']['gain_vs_distance_matched_pairs'], -0.2290, 1e-4)
chk4("§6.4 content N=64 matched", pm['content:N=64']['gain_vs_distance_matched_pairs'], 0.0092, 1e-4)
chk4("§6.4 k256 N=8", pm['lru:k=256:N=8']['gain_vs_distance_matched_pairs'], -0.0262, 1e-4)
chk4("§6.4 k256 N=32", pm['lru:k=256:N=32']['gain_vs_distance_matched_pairs'], -0.3096, 1e-4)
chk4("§6.4 k256 N=64", pm['lru:k=256:N=64']['gain_vs_distance_matched_pairs'], -0.6486, 1e-4)
chk4("§6.4 lastused N=8", pm['lastused:N=8']['gain_vs_distance_matched_pairs'], 0.0101, 1e-4)
chk4("§6.4 lastused N=32", pm['lastused:N=32']['gain_vs_distance_matched_pairs'], -0.0102, 1e-4)
chk4("§6.4 lastused N=64", pm['lastused:N=64']['gain_vs_distance_matched_pairs'], -0.0045, 1e-4)
# §7 正文限制
ctx = r['context_restricted_needle_leg']['rows']
for cid,v in [('distance:ctx:N=64',0.594971),('lru:k=64:ctx:N=64',0.578857),
              ('content:ctx:N=64',0.578857),('lru:k=256:ctx:N=64',0.546143)]:
    chk(f"§7 {cid}", ctx[cid]['recall'], v, 1e-5)
chk("§7 k256 unrestricted N=64", rows['lru:k=256:N=64']['retention']['recall'], 0.366699)
chk("§7 k256 cue broken n", rows['lru:k=256:N=64']['retention']['n_cue_broken'], 3682)
# §8 敏感性
sens = {s['id']:s for s in r['sensitivity']}
for cid,(dmg,rec) in [('lru:k=32:N=8',(0.187063,0.672852)),('lru:k=32:N=32',(0.712803,0.635742)),
   ('lru:k=32:N=64',(1.337398,0.574707)),('lru:k=64:N=8',(0.180614,0.682129)),('lru:k=64:N=32',(0.696677,0.647705)),
   ('lru:k=64:N=64',(1.330958,0.583008)),('lru:k=128:N=8',(0.176363,0.680176)),('lru:k=128:N=32',(0.681564,0.621094)),
   ('lru:k=128:N=64',(1.287473,0.555908)),('lru:k=256:N=8',(0.088516,0.619873)),('lru:k=256:N=32',(0.515204,0.49707)),
   ('lru:k=256:N=64',(1.122795,0.366699)),('lastused:N=8',(0.184766,0.659912)),('lastused:N=32',(0.649216,0.633789)),
   ('lastused:N=64',(1.250457,0.591553)),('content_keynorm:N=8',(0.318626,0.660889)),('content_keynorm:N=64',(1.461337,0.584717))]:
    src = sens[cid] if cid in sens else rows[cid]
    chk(f"§8 {cid} damage", src['damage_bpc'], dmg)
    chk(f"§8 {cid} recall", src['recall'] if cid in sens else src['retention']['recall'], rec)
# §9 成本
chk("§9 wall seconds", cost['wall_seconds'], 101.57, 1e-2)
chk("§9 dep wall", cost['dependency_wall_seconds'], 10.70, 1e-2)
chk("§9 needle wall", cost['needle_wall_seconds'], 32.15, 1e-2)
chk("§9 forwards", cost['n_forwards'], 4444, 0)
chk("§9 ms/fwd", cost['ms_per_forward'], 22.854, 1e-3)
chk("§9 peak", cost['peak_mem_mib'], 318.0, 1e-6)
chk("§9 windows", cost['n_windows'], 384)
chk("§9 needles", cost['n_needles_per_arm'], 4096)
chk("§9 configs", cost['n_configs'], 24)
chk("§9 pair curve", cost['n_configs_pair_curve'], 56)
chk("§9 ctx", cost['n_configs_context_restricted'], 12)
chk("§9 dep evals", cost['n_arm_evaluations_dependency_leg'], 80)
chk("§9 needle evals", cost['n_arm_evaluations_needle_leg'], 36)
chk("§9 reruns", cost['n_reruns'], 3)
chk("§9 log lines", cost['n_log_lines'], 39)
# 重叠度
ovl = {(o['N'],o['a'],o['b']):o for o in r['attention_quality']['overlap_deleted_sets']}
chk("§2 jac d-l N=8", ovl[(8,'distance:N=8','lru:k=64:N=8')]['mean_jaccard'], 0.033667, 1e-6)
chk("§2 jac d-c N=8", ovl[(8,'distance:N=8','content:N=8')]['mean_jaccard'], 0.059853, 1e-6)
chk("§2 jac d-c N=64", ovl[(64,'distance:N=64','content:N=64')]['mean_jaccard'], 0.755541, 1e-6)
chk("§2 jac d-l N=32", ovl[(32,'distance:N=32','lru:k=64:N=32')]['mean_jaccard'], 0.269646, 1e-6)
chk("§2 jac d-l N=64", ovl[(64,'distance:N=64','lru:k=64:N=64')]['mean_jaccard'], 0.472174, 1e-6)
# 噪声底
nfc = {c['id']+':'+c['leg']:c for c in r['noise_floor']['configs']}
chk("§noise dep d32", nfc['distance:N=32:dependency']['std_damage_bpc'], 0.004157, 1e-6)
chk("§noise dep l32", nfc['lru:k=64:N=32:dependency']['std_damage_bpc'], 0.004103, 1e-6)
chk("§noise dep c32", nfc['content:N=32:dependency']['std_damage_bpc'], 0.008274, 1e-6)
chk("§noise needle full", nfc['full:needle']['std_recall'], 0.029898, 1e-6)
chk("§noise needle d32", nfc['distance:N=32:needle']['std_recall'], 0.030278, 1e-6)
chk("§noise needle l32", nfc['lru:k=64:N=32:needle']['std_recall'], 0.024885, 1e-6)
chk("§noise needle c32", nfc['content:N=32:needle']['std_recall'], 0.03127, 1e-6)
er = r['criteria']['reconciliation_errata']
assert er['verdict_after_reconciliation'] == 'retracted_no_visible_benefit'; ok += 1
for cid,v in [('lru:k=64:N=8',-0.017),('content:N=8',0.013208),('lru:k=64:N=64',-0.229018),
              ('content:N=64',0.009197),('lastused:N=8',0.010133)]:
    chk(f"ERRATA {cid}", er['matched_pair_gains_vs_distance'][cid], v, 1e-6)
assert all(v < 0.014 for v in er['matched_pair_gains_vs_distance'].values()); ok += 1
print(f"\n所有 {ok} 项数字与 results.json 一致 ✓")
