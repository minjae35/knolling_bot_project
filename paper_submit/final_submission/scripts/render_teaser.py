"""Matched six-object Cycles teaser from immutable saved source/target poses.

Run with Blender, CPU only. --preview makes 768x512 renders; --check-only reopens
both final .blend files and independently compares all transforms and vertices.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
import render_publication as rp
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'renders';SOURCE=ROOT/'audit/evidence/full67_choice.json'
CAMERA={'location':[.3,-.38,.95],'target':[.3,.2,0.0],'ortho_scale_m':.64}
LIGHTS=[{'name':'Key','location':[-.12,-.20,.80],'energy':8.,'size':.42},
        {'name':'Fill','location':[.75,.40,.70],'energy':1.5,'size':.65},
        {'name':'Rim','location':[.15,.72,.55],'energy':3.,'size':.50}]

def setup(preview=False):
    scene=rp.setup(False,preview)
    for name in ['Tabletop']:
        ob=bpy.data.objects.get(name)
        if ob:bpy.data.objects.remove(ob,do_unlink=True)
    floor=bpy.data.objects['Studio floor'];floor.location.z=-.0002
    floor.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.86,.875,.89,1)
    cam=scene.camera;cam.location=CAMERA['location'];rp.aim(cam,CAMERA['target']);cam.data.ortho_scale=CAMERA['ortho_scale_m']
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.view_settings.exposure=0
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.22
    for cfg in LIGHTS:
        ob=bpy.data.objects[cfg['name']];ob.location=cfg['location'];ob.data.energy=cfg['energy'];ob.data.size=cfg['size'];rp.aim(ob,(.3,.2,0))
    return scene

def check_saved(data):
    checks={};largest=0
    for name,field in [('teaser6_before','source_pose'),('teaser6_after','target_pose')]:
        path=OUT/(name+'.blend');bpy.ops.wm.open_mainfile(filepath=str(path));s=bpy.context.scene
        objects={o.name:o for o in s.objects if o.name.startswith('o0')}
        assert set(objects)=={o['object_id'] for o in data['scene']['objects']}
        errors={};meshes={}
        for raw in data['scene']['objects']:
            ob=objects[raw['object_id']];p=raw[field]
            errors[ob.name]=max(abs(ob.location.x-p['x_m']),abs(ob.location.y-p['y_m']),abs(ob.location.z),abs(ob.rotation_euler.z-p['theta_rad']),abs(ob.rotation_euler.x),abs(ob.rotation_euler.y))
            a=raw['asset'];expected=[(x,y,z) for z in (0,a['h_m']) for x,y in a['footprint']]
            assert len(ob.data.vertices)==len(expected)
            meshes[ob.name]=max(abs(v.co[k]-q[k]) for v,q in zip(ob.data.vertices,expected) for k in range(3))
            assert errors[ob.name]<1e-7 and meshes[ob.name]<1e-7 and not ob.modifiers and tuple(ob.scale)==(1,1,1)
        largest=max(largest,max(errors.values()),max(meshes.values()))
        checks[name]={'status':'PASS','max_pose_error':max(errors.values()),'max_mesh_error':max(meshes.values()),
          'blend_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_sha256':hashlib.sha256((OUT/(name+'.png')).read_bytes()).hexdigest(),
          'camera_location':list(s.camera.location),'camera_euler':list(s.camera.rotation_euler),'ortho_scale':s.camera.data.ortho_scale,
          'resolution':[s.render.resolution_x,s.render.resolution_y],'engine':s.render.engine,'samples':s.cycles.samples,'device':s.cycles.device,'threads':s.render.threads,
          'view_transform':s.view_settings.view_transform,'look':s.view_settings.look,'exposure':s.view_settings.exposure,'denoising':s.cycles.use_denoising,
          'world_strength':s.world.node_tree.nodes['Background'].inputs['Strength'].default_value,
          'lights':{o.name:{'location':list(o.location),'rotation':list(o.rotation_euler),'energy':o.data.energy,'size':o.data.size} for o in s.objects if o.type=='LIGHT'},
          'materials':{o.name:{'base_color':list(o.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value),'roughness':o.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value} for o in objects.values()}}
    matching=['camera_location','camera_euler','ortho_scale','resolution','engine','samples','device','threads','view_transform','look','exposure','denoising','world_strength','lights','materials']
    for k in matching:assert checks['teaser6_before'][k]==checks['teaser6_after'][k],k
    assert checks['teaser6_before']['resolution']==[1920,1280]
    return checks,largest

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--preview',action='store_true');pa.add_argument('--check-only',action='store_true')
    args=pa.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    data=json.loads(SOURCE.read_text());assert data['task']=='PASS' and len(data['scene']['objects'])==6
    if not args.check_only:
        for name,field in [('teaser6_before','source_pose'),('teaser6_after','target_pose')]:
            s=setup(args.preview)
            for raw in data['scene']['objects']:rp.solid(raw,raw[field])
            filename=('preview_' if args.preview else '')+name+'.png';s.render.filepath=str(OUT/filename)
            if not args.preview:bpy.ops.wm.save_as_mainfile(filepath=str(OUT/(name+'.blend')),check_existing=False)
            bpy.ops.render.render(write_still=True);print('TEASER_RENDER_DONE',name,flush=True)
    if not args.preview:
        checks,largest=check_saved(data)
        report={'status':'PASS','sample_id':data['sample_id'],'task':data['task'],'instruction':data['instruction'],'blender_version':bpy.app.version_string,
          'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_path':str(SOURCE.relative_to(ROOT)),'camera':CAMERA,'lights':LIGHTS,
          'scope':'Presentation render of original project meshes, colors and material roughness at exact saved source and predicted target poses. Lighting/camera/tabletop are presentation choices. Not model input, new inference, or a benchmark.',
          'independent_check':'Reopen saved .blend scenes, compare all six object transforms and every mesh vertex directly to the frozen scene record; compare render/camera/light/material settings between before and after.',
          'maximum_absolute_pose_or_vertex_error':largest,'tolerance':1e-7,'renders':checks}
        (ROOT/'audit/teaser6_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
        print('TEASER_POSE_CHECK_PASS',largest,flush=True)
if __name__=='__main__':main()
