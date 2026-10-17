#!/usr/bin/env python3
"""Generate proposal-only Pokémon crop/reference pairs for root visual review.

Names are hand-read observations used only to narrow catalog candidates. Every
output remains PROPOSED; no identity or dataset label is approved by this tool.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dataset_cleanup/pokemon_reference_pairs'
CATALOG=ROOT/'runs/card_recognizer/pokemon_catalog_sync/catalog.json'
CATALOG_IMAGES=ROOT/'runs/card_recognizer/pokemon_catalog_sync'
MANIFESTS=[ROOT/f'runs/video_mining/binder-campaign/batch{i:02d}/crops/queries.json' for i in (1,2,3)]
# Indices refer to source-frame/crop manifest order (as viewed in contact sheets).
OBSERVED={
 'e34e86e2b6ff_frame_000000150_p0_det_001.png':('Dedenne','English name clearly visible'),
 'e34e86e2b6ff_frame_000000150_p0_det_003.png':('Revavroom ex','English name visible; ex suffix plausible'),
 'e34e86e2b6ff_frame_000000150_p0_det_004.png':('Kindler','English Trainer title/name visible'),
 'e34e86e2b6ff_frame_000000150_p0_det_006.png':('Numel','English name visible'),
 'e34e86e2b6ff_frame_000000150_p0_det_007.png':('Hala','English Trainer name visible'),
 'e34e86e2b6ff_frame_000000150_p0_det_008.png':('Zamazenta','Name begins Zamazenta; suffix unresolved'),
 'e34e86e2b6ff_frame_000000150_p0_det_009.png':('Talonflame','English name visible'),
 'e34e86e2b6ff_frame_000000150_p0_det_013.png':('Bea','English Trainer name visible'),
 'e34e86e2b6ff_frame_000000150_p0_det_020.png':('Mela','English Trainer name visible'),
 '8abb1ecd98e3_frame_000000330_p0_det_002.png':('Blastoise ex','English name and ex visible'),
 '8abb1ecd98e3_frame_000000330_p0_det_003.png':('Charmeleon','English name visible'),
 '8abb1ecd98e3_frame_000000330_p0_det_004.png':('Venusaur ex','name appears Venusaur; suffix tentative'),
 '8abb1ecd98e3_frame_000000330_p0_det_005.png':('Pikachu','English name visible'),
 '8abb1ecd98e3_frame_000000330_p0_det_007.png':('Squirtle','English name visible'),
 '8abb1ecd98e3_frame_000000330_p0_det_009.png':('Wartortle','English name visible'),
 '8abb1ecd98e3_frame_000000330_p0_det_010.png':('Charmander','English name visible'),
 '8abb1ecd98e3_frame_000000330_p0_det_011.png':('Venusaur','name visible; form suffix unresolved'),
 'ca534ce71558_deep-pocket-04_p0_det_000.png':('Psyduck','English name visible; collector number unreadable'),
}


def read_rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def image_match(ref,q):
    gray=lambda a:cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    sift=cv2.SIFT_create(nfeatures=1600)
    kr,dr=sift.detectAndCompute(gray(ref),None); kq,dq=sift.detectAndCompute(gray(q),None)
    base={'feature_count_reference':len(kr),'feature_count_query':len(kq),'good_match_count':0,'inlier_count':0,'inlier_ratio':0.0,'reference_coverage':0.0,'query_coverage':0.0,'reprojection_error_px':None,'mapped_quad_xy':None,'accepted_by_geometry_heuristic':False}
    if dr is None or dq is None:return base
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr,dq,k=2)
    good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
    base['good_match_count']=len(good)
    if len(good)<4:return base
    src=np.float32([kr[m.queryIdx].pt for m in good]).reshape(-1,1,2);dst=np.float32([kq[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.5)
    if H is None:return base
    ok=mask.ravel().astype(bool); a=src.reshape(-1,2)[ok];b=dst.reshape(-1,2)[ok]
    if not len(a):return base
    hull=lambda pts:cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))/max(1,ref.shape[0]*ref.shape[1])
    base.update(inlier_count=int(ok.sum()),inlier_ratio=float(ok.sum()/len(good)),reference_coverage=float(hull(a)),query_coverage=float(cv2.contourArea(cv2.convexHull(b.astype(np.float32)))/max(1,q.shape[0]*q.shape[1])))
    proj=cv2.perspectiveTransform(a.reshape(-1,1,2),H).reshape(-1,2)
    base['reprojection_error_px']=float(np.sqrt(np.mean(np.sum((proj-b)**2,axis=1))))
    h,w=ref.shape[:2]; quad=cv2.perspectiveTransform(np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2),H).reshape(4,2)
    base['mapped_quad_xy']=quad.tolist()
    area=abs(cv2.contourArea(quad.astype(np.float32)))/max(1,q.shape[0]*q.shape[1])
    ratio=np.mean([np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[3])])/max(1e-6,np.mean([np.linalg.norm(quad[2]-quad[1]),np.linalg.norm(quad[3]-quad[0])]))
    bounds=np.all(quad[:,0]>=-.03*q.shape[1]) and np.all(quad[:,0]<=1.03*q.shape[1]) and np.all(quad[:,1]>=-.03*q.shape[0]) and np.all(quad[:,1]<=1.03*q.shape[0])
    base['accepted_by_geometry_heuristic']=bool(ok.sum()>=12 and base['inlier_ratio']>=.45 and base['reference_coverage']>=.12 and base['query_coverage']>=.12 and base['reprojection_error_px']<=4 and .48<=ratio<=.82 and .18<=area<=1.1 and bounds)
    return base

def main():
    catalog=json.loads(CATALOG.read_text()); byname={}
    for c in catalog:
        if c.get('language')!='en' or c.get('game')!='pokemon' or c.get('domain')!='pokemon_physical_tcg' or not c.get('image_path'):continue
        ip=CATALOG_IMAGES/c['image_path']
        if ip.is_file():byname.setdefault(c.get('name','').casefold(),[]).append((c,ip))
    rows=[]
    for mp in MANIFESTS:
        if not mp.exists():continue
        data=json.loads(mp.read_text())
        for row in data['queries']:
            name_obs=OBSERVED.get(Path(row['image']).name)
            if not name_obs:continue
            name,evidence=name_obs; crop=mp.parent/row['image']
            candidates=byname.get(name.casefold(),[])
            ranked=[]
            try:q=read_rgb(crop)
            except Exception:continue
            for c,refpath in candidates:
                try:metrics=image_match(read_rgb(refpath),q)
                except Exception:continue
                score=(metrics['inlier_count'],metrics['inlier_ratio'],metrics['reference_coverage'],metrics['good_match_count'])
                ranked.append((score,c,refpath,metrics))
            ranked.sort(key=lambda x:x[0],reverse=True)
            best=ranked[:3]
            rows.append({'status':'PROPOSED_ONLY','root_visual_review_required':True,'observation':{'name_read':name,'evidence_note':evidence,'language_observed':'English'},
               'source_group':row['source_group'],'source_frame_image':row['source_frame_image'],'source_frame':row['source_frame'],'time_seconds':row['time_seconds'],
               'source_crop_path':str(crop.relative_to(ROOT)),'source_crop_sha256':row['sha256'],'source_crop_dimensions':[row['crop_width'],row['crop_height']],
               'crop_policy':row['crop_policy'],'identity_unverified':True,'candidate_references':[{'catalog_id':c.get('catalog_id'),'reference_path':str(rp.relative_to(ROOT)),'reference_sha256':sha(rp),'card_name':c.get('name'),'set_id':c.get('set_id'),'local_id':c.get('local_id'),'physical_finish':c.get('physical_finish'),'match_metrics':m,'suggested_rectification_quad_in_source_crop_xy':m.get('mapped_quad_xy')} for _,c,rp,m in best],
               'no_candidate_reason':None if best else 'no local reference with the observed English name'})
    OUT.mkdir(parents=True,exist_ok=True)
    # Best candidate contact sheet; item number matches proposal manifest index.
    sheet=Image.new('RGB',(1200, ((len(rows)+3)//4)*300),(248,248,248)); draw=ImageDraw.Draw(sheet)
    font=ImageFont.load_default()
    for i,r in enumerate(rows):
        col=i%4;rr=i//4;x=col*300;y=rr*300
        crop=ROOT/r['source_crop_path']; im=Image.open(crop).convert('RGB'); im.thumbnail((142,246));sheet.paste(im,(x+2,y+2))
        cand=r['candidate_references'][:1]
        if cand:
            ref=Image.open(ROOT/cand[0]['reference_path']).convert('RGB');ref.thumbnail((142,246));sheet.paste(ref,(x+150,y+2))
            cap=cand[0]['catalog_id']; met=cand[0]['match_metrics']; txt=f"#{i} {r['observation']['name_read']} -> {cap}\nInliers {met['inlier_count']} cov {met['reference_coverage']:.2f}/{met['query_coverage']:.2f}"
        else:txt=f"#{i} {r['observation']['name_read']} -> no match"
        draw.multiline_text((x+3,y+250),txt,fill='black',font=font,spacing=2)
    (OUT/'proposals.json').write_text(json.dumps({'schema_version':1,'status':'PROPOSED_ONLY','catalog_source':str(CATALOG.relative_to(ROOT)),'proposals':rows},indent=2)+'\n')
    sheet.save(OUT/'crop_reference_contact_sheet.jpg',quality=92)
    print(json.dumps({'proposal_count':len(rows),'output':str(OUT.relative_to(ROOT)),'proposals':str((OUT/'proposals.json').relative_to(ROOT)),'contact_sheet':str((OUT/'crop_reference_contact_sheet.jpg').relative_to(ROOT))},indent=2))
if __name__=='__main__':main()
