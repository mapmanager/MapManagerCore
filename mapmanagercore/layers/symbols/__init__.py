from shapely.geometry import Point, Polygon
import shapely.affinity


def cross(origin: Point, scale=1):
    x, y = origin.x, origin.y
    points = [
        (x - scale, y - scale / 4),
        (x - scale / 4, y - scale / 4),
        (x - scale / 4, y - scale),
        (x + scale / 4, y - scale),
        (x + scale / 4, y - scale / 4),
        (x + scale, y - scale / 4),
        (x + scale, y + scale / 4),
        (x + scale / 4, y + scale / 4),
        (x + scale / 4, y + scale),
        (x - scale / 4, y + scale),
        (x - scale / 4, y + scale / 4),
        (x - scale, y + scale / 4),
        (x - scale, y - scale / 4)
    ]

    return Polygon(points)


def xCross(origin: Point, scale=1):
    crossPoly = cross(origin, scale)
    return shapely.affinity.rotate(crossPoly, 45)
