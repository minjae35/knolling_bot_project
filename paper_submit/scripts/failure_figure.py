"""Saved failures explained by row tolerance and footprint/envelope geometry.

Consumes only bundled frozen evidence and completed Blender renders. No model
inference or pose modification. Guides use true oriented rectangular envelopes,
not the narrower visible chamfer/capsule silhouettes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, ConnectionPatch
from shapely.geometry import Polygon as SPolygon
from shapely.ops import nearest_points
import make_figures as m

R=m.ROOT/'renders'
H=294
EXTENT=(-.02,.62,.2-.64/3,.2+.64/3)
PREVIOUS_EXPORTS={
    'failure_explanation.pdf':'1b16a3cca596bad6b13b6747ca417f58924165cb970442773eb8d7c7491f3e80',
    'failure_explanation.svg':'ecba6987990ff7db3ec6931c9c4f57550a9e1aaf0a76a07e41a226d5ccb725be',
    'failure_explanation.png':'8e4b92f17fb3cf25e32478a9b457071bb3d587644b1cb572dfa91f8d1e72d437',
}


def frozen_figure6():
    """Check the user's protected figure without writing to any of its files."""
    frozen=json.loads((m.ROOT/'audit/figure6_frozen.json').read_text())
    status={name:hashlib.sha256((m.ROOT/name).read_bytes()).hexdigest()==digest
            for name,digest in frozen['sha256'].items()}
    assert all(status.values()),'Figure 6 or one of its source files changed'
    assert frozen['latex_figure_block'] in (m.ROOT/'appendix.tex').read_text()
    return status


def evidence():
    records=m.records()
    reports={r['mode']:r for r in json.loads((m.E/'v2_validation_smoke.json').read_text())}
    scenes={}
    for mode in ('full','partial'):
        report=reports[mode]
        objects={o['object_id']:o for o in records[report['sample_id']]['source']['objects']}
        poses={p['object_id']:p for p in report['objects']}
        polygons={k:SPolygon(m.envelope(objects[k],p)) for k,p in poses.items()}
        scenes[mode]=(objects,poses,polygons)
    objects,poses,polygons=scenes['full']
    rows=[]
    for p in sorted(poses.values(),key=lambda p:(-p['y_m'],p['x_m'])):
        if not rows or rows[-1][0]['y_m']-p['y_m']>.0101:rows.append([])
        rows[-1].append(p)
    gaps=[]
    for above,below in zip(rows,rows[1:]):
        bottom=min(polygons[p['object_id']].bounds[1] for p in above)
        top=max(polygons[p['object_id']].bounds[3] for p in below)
        gaps.append({'actual_mm':1000*(bottom-top),'upper_bottom_y_m':bottom,'lower_top_y_m':top,
                     'above_ids':[p['object_id'] for p in above],'below_ids':[p['object_id'] for p in below]})
    gap=max(gaps,key=lambda z:abs(z['actual_mm']-15))
    objects,poses,polygons=scenes['partial']
    pairs=[(polygons[a].distance(polygons[b]),a,b) for a in polygons for b in polygons if a<b]
    distance,a,b=min(pairs)
    near=nearest_points(polygons[a],polygons[b])
    result={'full':dict(gap,sample_id=reports['full']['sample_id'],allowed_band_mm=[13,17],
                        excess_over_upper_limit_mm=gap['actual_mm']-17,
                        guide_world_xy_m=[[.175,gap['upper_bottom_y_m']],[.175,gap['lower_top_y_m']]]),
            'partial':{'sample_id':reports['partial']['sample_id'],'closest_pair':[a,b],
                       'actual_clearance_mm':distance*1000,'nominal_minimum_mm':10.,'numerical_tolerance_mm':.1,
                       'effective_minimum_mm':9.9,'deficit_um':(9.9-distance*1000)*1000,
                       'deficit_native_pixels':(9.9-distance*1000)/(.64*1000/1920),
                       'nearest_points_world_xy_m':[[p.x,p.y] for p in near]},
            'provenance':{'method':'Geometry recomputed from frozen validation poses and original rectangular envelopes. These are illustrative saved predictions, not a new benchmark.',
                          'overhead_extent_world_m':EXTENT,
                          'input_sha256':{str(p.relative_to(m.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [m.E/'v2_validation_smoke.json',m.E/'v2_smoke_source_records.json',R/'smoke_full_overhead.png',R/'smoke_partial_overhead.png']}}}
    assert result['full']['excess_over_upper_limit_mm']>0
    assert 3.8<result['partial']['deficit_um']<4.
    return result,scenes


def footprint_analysis(values,scenes):
    """Read-only planar analysis of the original, extruded mesh footprints."""
    result={'scope':'Exact planar geometry from saved predicted poses and original footprint vertices. This is not a new model evaluation or a physical execution claim.',
      'source_sha256':values['provenance']['input_sha256'],
      'scored_geometry':'Oriented major/minor-dimension rectangular envelopes',
      'footprint_geometry':'Original project polygon footprints transformed to saved predicted poses',
      'modes':{}}
    for mode,(objects,poses,envelopes) in scenes.items():
        footprints={oid:SPolygon(m.apply_pose(objects[oid]['asset']['footprint'],poses[oid])) for oid in poses}
        pairs=[]
        for a in sorted(poses):
            for b in sorted(poses):
                if a>=b:continue
                fp=nearest_points(footprints[a],footprints[b]);ep=nearest_points(envelopes[a],envelopes[b])
                pairs.append({'pair':[a,b],
                  'footprint_clearance_mm':1000*footprints[a].distance(footprints[b]),
                  'envelope_clearance_mm':1000*envelopes[a].distance(envelopes[b]),
                  'footprint_nearest_points_m':[[v.x,v.y] for v in fp],
                  'envelope_nearest_points_m':[[v.x,v.y] for v in ep]})
        result['modes'][mode]={'all_pairs':pairs,
          'minimum_footprint':min(pairs,key=lambda x:x['footprint_clearance_mm']),
          'minimum_envelope':min(pairs,key=lambda x:x['envelope_clearance_mm'])}
    selected=result['modes']['partial']['minimum_envelope']
    assert selected['pair']==values['partial']['closest_pair']
    assert abs(selected['envelope_clearance_mm']-values['partial']['actual_clearance_mm'])<1e-10
    assert selected['pair']==result['modes']['partial']['minimum_footprint']['pair']
    assert 12.819 < selected['footprint_clearance_mm'] < 12.821
    result['partial_example_conclusion']={
       'pair':selected['pair'],'footprint_clearance_mm':selected['footprint_clearance_mm'],
       'scored_envelope_clearance_mm':selected['envelope_clearance_mm'],
       'effective_clearance_minimum_mm':9.9,
       'interpretation':'The visible footprint separation exceeds the clearance threshold, but the conservative scored envelopes miss it by 3.897 micrometres. The underlying poses are identical in both representations.'}
    return result


def generate():
    from figure_tex import compile_figure,node,picture
    protected_before=frozen_figure6()
    annotation_path=m.ROOT/'audit/failure_annotations.json'
    original=annotation_path.read_bytes()
    values,scenes=evidence()
    assert original==(json.dumps(values,indent=2)+'\n').encode()
    analysis=footprint_analysis(values,scenes)
    analysis_path=m.ROOT/'audit/footprint_clearance_analysis.json'
    assert analysis_path.read_bytes()==(json.dumps(analysis,indent=2)+'\n').encode()
    detail_provenance=json.loads((m.ROOT/'audit/detail_render_provenance.json').read_text())
    context_width=147.0
    detail_width=210.0
    gutter=24.0
    group_left=(396-context_width-gutter-detail_width)/2
    detail_left=group_left+context_width+gutter
    detail_scale=detail_width/272.0
    row_heights={'full':102*detail_scale,'partial':118*detail_scale}
    title_a,sub_a,row_a=11.0,29.0,44.0
    foot_a=row_a+row_heights['full']+11.0
    title_b,sub_b=foot_a+16.0,foot_a+32.0
    row_b=sub_b+15.0
    foot_b=row_b+row_heights['partial']+11.0
    row_tops={'full':row_a,'partial':row_b}
    assert foot_b+8<=H,(foot_b,H)
    body=[]
    def text(x,y,label,options='anchor=west'):
        body.append(node(x,y,label,options))
    def line(points,options='draw=ink,line width=.45bp'):
        body.append('\\draw['+options+'] '+' -- '.join(f'({x:.7f},{y:.7f})' for x,y in points)+';\n')
    def mapping(x,y,w,limits):
        xmin,xmax,ymin,ymax=limits;scale=w/(xmax-xmin)
        return lambda px,py:np.array([x+(px-xmin)*scale,y+(ymax-py)*scale])
    def crop_image(name,x,y,w,h,limits):
        f=mapping(x,y,w,limits);scale=w/(limits[1]-limits[0])
        assert abs(h/(limits[3]-limits[2])-scale)<1e-8
        ix,iy=f(EXTENT[0],EXTENT[3])
        body.append(r'\begin{scope}'+'\n')
        body.append(r'\clip '+f'({x},{y}) rectangle ({x+w},{y+h});\n')
        body.append(picture(f'renders/{name}.png',ix,iy,(EXTENT[1]-EXTENT[0])*scale,(EXTENT[3]-EXTENT[2])*scale))
        body.append(r'\end{scope}'+'\n')
        return f
    def zoom_correspondence(transform,limits,x,y,w,h):
        """Match the exact world crop with its printed inset, without resampling."""
        a=transform(limits[0],limits[3]);b=transform(limits[1],limits[2])
        corners=[a,[b[0],a[1]],b,[a[0],b[1]],a]
        # A continuous, unfilled frame distinguishes the crop from dashed
        # scored envelopes. Quiet leaders carry correspondence across the gutter.
        leaders=[]
        for start,end in [([b[0],a[1]],[x,y]),(b,[x,y+h])]:
            # Exit the overview horizontally before crossing the gutter;
            # this keeps the partial upper leader clear of the green object.
            points=[start,[group_left+context_width+2,start[1]],end]
            line(points,'draw=accent!55!white,line width=.3bp')
            leaders.append([[float(px),float(py)] for px,py in points])
        line(corners,'draw=white,line width=1.15bp')
        line(corners,'draw=accent,line width=.45bp')
        line([[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]],
             'draw=accent,line width=.45bp')
        scale_x=w/(b[0]-a[0]);scale_y=h/(b[1]-a[1])
        assert abs(scale_x-scale_y)<1e-10
        text(x,y-15,rf'Detail\enspace ${scale_x:.2f}\times$',
             r'anchor=west,text=accent,font=\fontsize{10}{12}\selectfont')
        return {'linear_magnification':scale_x,'display_label':f'{scale_x:.2f}x',
                'roi_print_bounds_bp':[float(a[0]),float(a[1]),float(b[0]),float(b[1])],
                'detail_print_bounds_bp':[x,y,x+w,y+h],
                'leader_paths_bp':leaders}
    def detail(name,x,y,w,h):
        record=detail_provenance['renders'][name]
        path=m.ROOT/'renders'/record['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['image_sha256']
        assert abs(record['resolution'][0]/record['resolution'][1]-w/h)<1e-10
        body.append(picture(f'renders/{record["image"]}',x,y,w,h))
        return mapping(x,y,w,record['world_extent_m']),record['world_extent_m']
    def overlay(points,style):
        line(points,'draw=white,line width=1.5bp')
        line(points,style)

    text(group_left,title_a,r'\textbf{a}\quad Row-gap violation',r'anchor=west,font=\fontsize{11}{13}\selectfont')
    text(group_left,title_b,r'\textbf{b}\quad Envelope-based clearance',r'anchor=west,font=\fontsize{11}{13}\selectfont')
    # Match each overview's top/bottom edges to its associated detail. The
    # horizontal world spans and printed widths are unchanged, so object scale
    # stays fixed. Center overview crops vertically on the relevant objects.
    full_objects,full_poses,_=scenes['full']
    full_points=np.array([q for oid,pose in full_poses.items()
        for q in m.apply_pose(full_objects[oid]['asset']['footprint'],pose)])
    full_mid=(full_points[:,1].min()+full_points[:,1].max())/2
    full_half=row_heights['full']*(.49-.13)/context_width/2
    partial_polygons=scenes['partial'][2]
    partial_pair=values['partial']['closest_pair']
    partial_mid=(min(partial_polygons[k].bounds[1] for k in partial_pair)
                 +max(partial_polygons[k].bounds[3] for k in partial_pair))/2
    partial_half=row_heights['partial']*(.39-.08)/context_width/2
    full_context=(.13,.49,full_mid-full_half,full_mid+full_half)
    partial_context=(.08,.39,partial_mid-partial_half,partial_mid+partial_half)
    ca=crop_image('smoke_full_overhead',group_left,row_tops['full'],context_width,row_heights['full'],full_context)
    cb=crop_image('smoke_partial_overhead',group_left,row_tops['partial'],context_width,row_heights['partial'],partial_context)
    ga,full_detail=detail('detail_row_gap',detail_left,row_tops['full'],detail_width,row_heights['full'])
    gb,partial_detail=detail('detail_clearance',detail_left,row_tops['partial'],detail_width,row_heights['partial'])
    zoom_links={
        'full':zoom_correspondence(ca,full_detail,detail_left,row_tops['full'],detail_width,row_heights['full']),
        'partial':zoom_correspondence(cb,partial_detail,detail_left,row_tops['partial'],detail_width,row_heights['partial'])}
    text(group_left,sub_a,'Full prediction',r'anchor=west,text=muted,font=\fontsize{10}{12}\selectfont')
    text(group_left,sub_b,'Partial prediction',r'anchor=west,text=muted,font=\fontsize{10}{12}\selectfont')

    # Only the two row edges defining the score are marked, in the rendered
    # image coordinate frame. No silhouettes or tolerance band are redrawn.
    g=values['full'];bottom=g['upper_bottom_y_m'];top=g['lower_top_y_m']
    overlay([ga(.218,bottom),ga(.266,bottom)],'draw=ink,line width=.4bp,dash pattern=on 2bp off 1.5bp')
    overlay([ga(.258,top),ga(.303,top)],'draw=ink,line width=.4bp,dash pattern=on 2bp off 1.5bp')
    aa,bb=ga(.262,bottom),ga(.262,top)
    overlay([aa,bb],'draw=signal,line width=.6bp')
    for p in (aa,bb):overlay([p+[-2,0],p+[2,0]],'draw=signal,line width=.6bp')
    text(*ga(.253,(bottom+top)/2),rf'${g["actual_mm"]:.3f}\,\mathrm{{mm}}$',
         r'anchor=east,text=signal,fill=white,inner sep=1.2bp,font=\fontsize{10}{12}\selectfont')
    text(detail_left,foot_a,r'Allowed 13--17 mm; excess $0.657$ mm',
         r'anchor=west,text=signal,font=\fontsize{10}{12}\selectfont')

    # The actual object render supplies the footprint. Scored rectangles and
    # nearest-point segments are the only added geometry.
    objects,poses,_=scenes['partial'];p=values['partial'];pair=p['closest_pair']
    item=analysis['modes']['partial']['minimum_envelope']
    for oid in pair:
        points=[gb(*q) for q in m.envelope(objects[oid],poses[oid])]
        points.append(points[0])
        line(points,'draw=white,line width=1.2bp,dash pattern=on 2bp off 1.5bp')
        line(points,'draw=ink,line width=.4bp,dash pattern=on 2bp off 1.5bp')
    env=np.array(item['envelope_nearest_points_m']);fp=np.array(item['footprint_nearest_points_m'])
    for segment,col,label,offset in [(env,'signal','env',-9),(fp,'teal!85!black','fp',9.5)]:
        points=[gb(*v) for v in segment]
        overlay(points,f'draw={col},line width=.75bp')
        for xx,yy in points:
            body.append(f'\\draw[draw={col},fill=white,line width=.45bp] ({xx:.7f},{yy:.7f}) circle[radius=1bp];\n')
        center=gb(*segment.mean(axis=0));center[1]+=offset
        text(*center,rf'$\mathrm{{{label}}}$',
             f'anchor=center,text={col},fill=white,inner sep=1bp,font=\\fontsize{{10}}{{12}}\\selectfont')
    text(detail_left,foot_b,rf'Envelope ${item["envelope_clearance_mm"]:.4f}<9.9$ mm',
         r'anchor=west,text=signal,font=\fontsize{10}{12}\selectfont')
    text(detail_left+detail_width,foot_b,rf'Footprint {item["footprint_clearance_mm"]:.3f} mm',
         r'anchor=east,text=teal!85!black,font=\fontsize{10}{12}\selectfont')
    render=compile_figure('failure_explanation',''.join(body),H,label_pt=10)
    protected_after=frozen_figure6()
    record={'revision':'Larger panels and 10–11 pt labels, still within the appendix page',
       'physical_size_pt':[396,H],
       'design':'The 312 bp image group is centered with equal 42 bp outer margins and a 40 bp gutter. Each overview matches its detail top and bottom edge through vertical reframing at unchanged image scale. The full crop is centered on all objects, the partial crop on the scored pair; an unrelated previously clipped corner fragment lies outside the new partial crop. Both image labels share a header baseline; metrics stay within the detail column. Matched slate-blue crop frames, fine leaders and calibrated magnification are retained.',
       'zoom_correspondence':zoom_links,
       'source_sha256':values['provenance']['input_sha256'],
       'annotations_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
       'footprint_analysis_sha256':hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
       'detail_render_manifest':'audit/detail_render_provenance.json',
       'detail_images':{name:detail_provenance['renders'][name]['image_sha256'] for name in ['detail_row_gap','detail_clearance']},
       'closeup_layout':{'width_bp':detail_width,'previous_width_bp':160,'original_width_bp':272,
           'linear_scale_ratio':detail_scale,'object_area_ratio':detail_scale**2,
           'left_bp':detail_left,'row_tops_bp':row_tops,'row_heights_bp':row_heights,
           'figure_height_bp':H,'previous_figure_height_bp':240,
           'context_width_bp':context_width,'outer_margins_bp':[group_left,group_left],
           'gutter_bp':gutter,'label_pt':10,'panel_title_pt':11,
           'source_images_and_detail_world_crops_unchanged':True,
           'overview_vertical_reframing_without_rescaling':True},
       'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'export_sha256':{f'failure_explanation.{ext}':digest for ext,digest in render['exports_sha256'].items()},
       'crop_world_xy_limits_m':{'full_context':full_context,'full_detail':full_detail,'partial_context':partial_context,'partial_detail':partial_detail},
       'partial_distances_mm':{'footprint':item['footprint_clearance_mm'],'envelope':item['envelope_clearance_mm'],'effective_minimum':9.9},
       'full_gap_mm':g['actual_mm'],'full_gap_allowed_mm':[13,17],
       'text_bounds':'PASS','annotation_bytes_unchanged':original==annotation_path.read_bytes(),
       'figure6_before':protected_before,'figure6_after':protected_after,'new_inference':False,'original_poses_or_render_files_modified':False,
       'editable_source':'figure_sources/failure_explanation.tex'}
    (m.ROOT/'audit/failure_reference_revision.json').write_text(json.dumps(record,indent=2)+'\n')
    return values

if __name__=='__main__':generate()
