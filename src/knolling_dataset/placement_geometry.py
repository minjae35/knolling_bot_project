"""Conservative SAT placement checks; independent validator uses Shapely.

A separating-axis gap >= g is sufficient for Euclidean clearance >= g.
It can reject some valid diagonal placements, which is acceptable for sampling.
"""
import math
from .schema import WIDTH_M, HEIGHT_M


def corners(asset, pose):
    c,s = math.cos(pose.theta_rad),math.sin(pose.theta_rad)
    return tuple((pose.x_m+c*x-s*y,pose.y_m+s*x+c*y)
                 for x,y in ((-asset.a_m/2,-asset.b_m/2),(asset.a_m/2,-asset.b_m/2),
                             (asset.a_m/2,asset.b_m/2),(-asset.a_m/2,asset.b_m/2)))


def bounds(asset,pose):
    points = corners(asset,pose)
    return min(x for x,y in points),min(y for x,y in points),max(x for x,y in points),max(y for x,y in points)


def inside(asset,pose,region):
    l,b,r,t = bounds(asset,pose)
    return l >= region[0]-1e-9 and b >= region[1]-1e-9 and r <= region[2]+1e-9 and t <= region[3]+1e-9


def separated(asset_a,pose_a,asset_b,pose_b,gap):
    pa,pb = corners(asset_a,pose_a),corners(asset_b,pose_b)
    for theta in (pose_a.theta_rad,pose_b.theta_rad):
        for angle in (theta,theta+math.pi/2):
            c,s = math.cos(angle),math.sin(angle)
            a,b = [x*c+y*s for x,y in pa],[x*c+y*s for x,y in pb]
            if max(min(b)-max(a),min(a)-max(b)) >= gap-1e-9:
                return True
    return False


def safe_placement(asset,pose,placed,gap=.010,margin=.010):
    return inside(asset,pose,(margin,margin,WIDTH_M-margin,HEIGHT_M-margin)) and all(
        separated(asset,pose,other_asset,other_pose,gap) for other_asset,other_pose in placed)
