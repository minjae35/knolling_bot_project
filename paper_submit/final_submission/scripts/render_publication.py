"""Publication Cycles renders from frozen scene records, without model inference.

Run: blender --background --threads 8 --python scripts/render_publication.py -- [--preview]
Only the presentation camera, lighting and tabletop differ from dataset renders.
Object footprints, heights, source poses and predicted target poses are immutable.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'audit/evidence'; OUT=ROOT/'renders'
PALETTE={'red':(.62,.025,.025,1),'green':(.025,.42,.045,1),'blue':(.025,.07,.62,1)}
CAMERA={'location':[.3,-.54,.95],'target':[.3,.2,0.0],'ortho_scale_m':.70}
OVERHEAD={'location':[.3,.2,1.0],'target':[.3,.2,0.0],'ortho_scale_m':.64}
LIGHTS=[{'name':'Key','location':[-.12,-.25,.85],'energy':7,'size':.55},
        {'name':'Fill','location':[.75,.32,.65],'energy':2,'size':.65},
        {'name':'Rim','location':[.15,.72,.55],'energy':3,'size':.45}]


def material(name,color,roughness=.65):
    mat=bpy.data.materials.new(name);mat.diffuse_color=color;mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=color
    bsdf.inputs['Roughness'].default_value=roughness
    return mat


def solid(raw,pose):
    a=raw['asset'];pts=a['footprint'];n=len(pts)
    verts=[(x,y,z) for z in (0,a['h_m']) for x,y in pts]
    faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]
    faces += [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
    mesh=bpy.data.meshes.new(raw['object_id']);mesh.from_pydata(verts,[],faces);mesh.update()
    obj=bpy.data.objects.new(raw['object_id'],mesh);bpy.context.collection.objects.link(obj)
    obj.location=(pose['x_m'],pose['y_m'],0);obj.rotation_euler=(0,0,pose['theta_rad'])
    obj.data.materials.append(material(raw['object_id']+'_mat',PALETTE[raw['color']]))
    # Exact polygonal project asset, without beveling or moving vertices.
    return obj


def aim(obj,target):obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def setup(overhead=False,preview=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32 if preview else 128
    scene.cycles.use_denoising=True;scene.cycles.seed=20261115;scene.cycles.use_animated_seed=False
    scene.render.threads_mode='FIXED';scene.render.threads=8
    scene.render.resolution_x=768 if preview else 1920
    scene.render.resolution_y=512 if preview else 1280
    scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
    scene.render.film_transparent=False;scene.render.dither_intensity=0
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=0;scene.view_settings.gamma=1
    world=bpy.data.worlds.new('World');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.30;scene.world=world
    cfg=OVERHEAD if overhead else CAMERA
    cam=bpy.data.objects.new('Camera',bpy.data.cameras.new('Camera'));bpy.context.collection.objects.link(cam)
    cam.location=cfg['location'];aim(cam,cfg['target']);cam.data.type='ORTHO';cam.data.ortho_scale=cfg['ortho_scale_m'];cam.data.clip_start=.01;cam.data.clip_end=10;scene.camera=cam
    # Ground and a neutral tabletop; the usable 0.60 x 0.40 workspace is unchanged.
    bpy.ops.mesh.primitive_plane_add(size=200,location=(.3,.2,-.016));bpy.context.object.name='Studio floor';bpy.context.object.data.materials.append(material('studio',(.94,.95,.96,1),.8))
    bpy.ops.mesh.primitive_cube_add(size=1,location=(.3,.2,-.008));table=bpy.context.object;table.name='Tabletop';table.dimensions=(.64,.44,.016)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    table.data.materials.append(material('table',(.94,.95,.96,1),.72))
    bevel=table.modifiers.new('Table edge only','BEVEL');bevel.width=.003;bevel.segments=4
    table.modifiers.new('Table normals','WEIGHTED_NORMAL')
    for cfg in LIGHTS:
        light=bpy.data.objects.new(cfg['name'],bpy.data.lights.new(cfg['name'],'AREA'));bpy.context.collection.objects.link(light)
        light.location=cfg['location'];light.data.energy=cfg['energy'];light.data.shape='DISK';light.data.size=cfg['size'];aim(light,(.3,.2,0))
    return scene


def jobs():
    full=json.loads((E/'full8b_choice.json').read_text());objs=full['scene']['objects']
    result=[('full_before',objs,{o['object_id']:o['source_pose'] for o in objs},full['sample_id'],False),
            ('full_after',objs,{o['object_id']:o['target_pose'] for o in objs},full['sample_id'],False)]
    records=json.loads((E/'v2_smoke_source_records.json').read_text());smoke=json.loads((E/'v2_validation_smoke.json').read_text())
    first=records[smoke[0]['sample_id']]['source']['objects']
    result += [('smoke_before',first,{o['object_id']:o['source_pose'] for o in first},smoke[0]['sample_id'],False)]
    for r in smoke:
        objs=records[r['sample_id']]['source']['objects'];poses={p['object_id']:{k:p[k] for k in ['x_m','y_m','theta_rad']} for p in r['objects']}
        result.append(('smoke_'+r['mode'],objs,poses,r['sample_id'],False))
        if r['mode'] in ('full','partial'):result.append(('smoke_'+r['mode']+'_overhead',objs,poses,r['sample_id'],True))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--only',default='')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(exist_ok=True)
    provenance={'blender_version':bpy.app.version_string,'renderer':'Cycles CPU','threads':8,'samples':128,'max_samples':128,'adaptive_sampling':True,'denoising':True,
                'color_management':'AgX / Medium High Contrast','object_palette':PALETTE,'roughness':.65,
                'oblique_camera':CAMERA,'overhead_camera':OVERHEAD,'lights':LIGHTS,
                'scene_policy':'Exact original footprint extrusion and heights; source or stored predicted poses unchanged. Only presentation camera, light and tabletop change. These re-renders are not model input images.',
                'input_sha256':{n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['full8b_choice.json','v2_validation_smoke.json','v2_smoke_source_records.json']},'renders':{}}
    for name,objs,poses,sid,overhead in jobs():
        if args.only and name not in args.only.split(','):continue
        scene=setup(overhead,args.preview)
        for obj in objs:solid(obj,poses[obj['object_id']])
        filename=('preview_'+name if args.preview else name)+'.png';scene.render.filepath=str(OUT/filename)
        if not args.preview:bpy.ops.wm.save_as_mainfile(filepath=str(OUT/(name+'.blend')),check_existing=False)
        bpy.ops.render.render(write_still=True)
        provenance['renders'][name]={'sample_id':sid,'image':filename,'camera':'overhead' if overhead else 'oblique',
            'resolution':[scene.render.resolution_x,scene.render.resolution_y],'poses':poses,
            'sha256':hashlib.sha256((OUT/filename).read_bytes()).hexdigest()}
        (OUT/('preview_metadata.json' if args.preview else 'render_metadata.json')).write_text(json.dumps(provenance,indent=2)+'\n')
        print('PAPER_RENDER_DONE '+name,flush=True)
    if not args.preview:(ROOT/'audit/blender_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()
