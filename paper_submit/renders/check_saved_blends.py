"""Independent read-only check of saved Blender files against frozen pose evidence."""
from pathlib import Path
import json
import math
import hashlib
import bpy

ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'audit/evidence'; OUT=ROOT/'renders'
full=json.loads((E/'full8b_choice.json').read_text())
smoke=json.loads((E/'v2_validation_smoke.json').read_text())
records=json.loads((E/'v2_smoke_source_records.json').read_text())
expect={
    'full_before':(full['scene']['objects'],{o['object_id']:o['source_pose'] for o in full['scene']['objects']}),
    'full_after':(full['scene']['objects'],{o['object_id']:o['target_pose'] for o in full['scene']['objects']}),
}
first=records[smoke[0]['sample_id']]['source']['objects']
expect['smoke_before']=(first,{o['object_id']:o['source_pose'] for o in first})
for report in smoke:
    value=(records[report['sample_id']]['source']['objects'],{o['object_id']:o for o in report['objects']})
    expect['smoke_'+report['mode']]=value
    if report['mode'] in ('full','partial'):expect['smoke_'+report['mode']+'_overhead']=value
checks={};maximum=0
for name,(objects,poses) in expect.items():
    path=OUT/(name+'.blend');bpy.ops.wm.open_mainfile(filepath=str(path))
    scene=bpy.context.scene;found={o.name:o for o in scene.objects if o.name in poses}
    assert set(found)==set(poses),(name,'object identity')
    errors={};mesh_errors={}
    for raw in objects:
        oid=raw['object_id'];ob=found[oid];p=poses[oid]
        errors[oid]={'x_m':abs(ob.location.x-p['x_m']),'y_m':abs(ob.location.y-p['y_m']),
                     'theta_rad':abs(ob.rotation_euler.z-p['theta_rad']),
                     'z_m':abs(ob.location.z),'tilt_x_rad':abs(ob.rotation_euler.x),'tilt_y_rad':abs(ob.rotation_euler.y)}
        assert max(errors[oid].values())<=1e-7,(name,oid,errors[oid])
        assert tuple(ob.scale)==(1,1,1),(name,oid,'scale')
        asset=raw['asset'];expected=[(x,y,z) for z in (0,asset['h_m']) for x,y in asset['footprint']]
        assert len(ob.data.vertices)==len(expected),(name,oid,'mesh vertex count')
        mesh_errors[oid]=max(abs(float(v.co[k])-p[k]) for v,p in zip(ob.data.vertices,expected) for k in range(3))
        assert mesh_errors[oid]<=1e-7,(name,oid,'mesh')
        assert not ob.modifiers,(name,oid,'modified mesh')
    highest=max(v for e in errors.values() for v in e.values());maximum=max(maximum,highest)
    assert scene.render.resolution_x==1920 and scene.render.resolution_y==1280
    assert scene.cycles.samples==128 and scene.render.engine=='CYCLES'
    checks[name]={'status':'PASS','max_pose_absolute_error':highest,'errors':errors,'max_mesh_vertex_error_m':max(mesh_errors.values()),
                  'blend_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'camera_location':list(scene.camera.location),'camera_euler':list(scene.camera.rotation_euler),
                  'ortho_scale_m':scene.camera.data.ortho_scale,
                  'resolution':[scene.render.resolution_x,scene.render.resolution_y],
                  'samples':scene.cycles.samples,'adaptive_sampling':scene.cycles.use_adaptive_sampling,'denoising':scene.cycles.use_denoising,
                  'world_strength':scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value,
                  'view_transform':scene.view_settings.view_transform,'look':scene.view_settings.look,
                  'object_materials':{o.name:{'base_color':list(o.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value),
                  'roughness':o.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value} for o in found.values()},
                  'lights':{o.name:{'location':list(o.location),'energy':o.data.energy,'size':o.data.size} for o in scene.objects if o.type=='LIGHT'}}
summary={'status':'PASS','method':'Open each saved .blend and compare object transforms and every mesh vertex independently to frozen evidence; no rendering or inference.',
         'tolerance_absolute':1e-7,'maximum_pose_error':maximum,'note':'Residuals are Blender float32 storage quantization; there are no layout adjustments.',
         'renders':checks}
(ROOT/'audit/blender_pose_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print('BLENDER_POSE_AUDIT_PASS',maximum,flush=True)
