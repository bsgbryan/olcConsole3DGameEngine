#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <fstream>
#include <strstream>
#include <algorithm>

using namespace std;

struct vector3d {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct triangle {
    vector3d p[3];
    olc::Pixel color;
};

struct mesh {
    vector<triangle> triangles;
};

struct matrix4x4 {
    float m[4][4] = { 0 };
};

class olcConsoleEngine3D : public olc::PixelGameEngine {
public:
    olcConsoleEngine3D() {
        sAppName = "Demo";
    }

private:
    mesh cube;
    matrix4x4 projection;
    vector3d camera;
    vector3d lookDirection;
    float theta;
    float yaw;

    vector3d Matrix_MultiplyVector(matrix4x4 &m, vector3d &i) {
        vector3d v;

        v.x = i.x * m.m[0][0] + i.y * m.m[1][0] + i.z * m.m[2][0] + i.w * m.m[3][0];
        v.y = i.x * m.m[0][1] + i.y * m.m[1][1] + i.z * m.m[2][1] + i.w * m.m[3][1];
        v.z = i.x * m.m[0][2] + i.y * m.m[1][2] + i.z * m.m[2][2] + i.w * m.m[3][2];
        v.w = i.x * m.m[0][3] + i.y * m.m[1][3] + i.z * m.m[2][3] + i.w * m.m[3][3];

        return v;
    }

    matrix4x4 Matrix_MakeIdentity() {
        matrix4x4 matrix;

        matrix.m[0][0] = 1.0f;
        matrix.m[1][1] = 1.0f;
        matrix.m[2][2] = 1.0f;
        matrix.m[3][3] = 1.0f;

        return matrix;
    }

    matrix4x4 Matrix_MakeRotationX(float fAngleRad) {
        matrix4x4 matrix;

        matrix.m[0][0] = 1.0f;
        matrix.m[1][1] = cosf(fAngleRad);
        matrix.m[1][2] = sinf(fAngleRad);
        matrix.m[2][1] = -sinf(fAngleRad);
        matrix.m[2][2] = cosf(fAngleRad);
        matrix.m[3][3] = 1.0f;

        return matrix;
    }

    matrix4x4 Matrix_MakeRotationY(float fAngleRad) {
        matrix4x4 matrix;

        matrix.m[0][0] = cosf(fAngleRad);
        matrix.m[0][2] = sinf(fAngleRad);
        matrix.m[2][0] = -sinf(fAngleRad);
        matrix.m[1][1] = 1.0f;
        matrix.m[2][2] = cosf(fAngleRad);
        matrix.m[3][3] = 1.0f;

        return matrix;
    }

    matrix4x4 Matrix_MakeRotationZ(float fAngleRad) {
        matrix4x4 matrix;

        matrix.m[0][0] = cosf(fAngleRad);
        matrix.m[0][1] = sinf(fAngleRad);
        matrix.m[1][0] = -sinf(fAngleRad);
        matrix.m[1][1] = cosf(fAngleRad);
        matrix.m[2][2] = 1.0f;
        matrix.m[3][3] = 1.0f;

        return matrix;
    }

    matrix4x4 Matrix_MakeTranslation(float x, float y, float z) {
        matrix4x4 matrix;

        matrix.m[0][0] = 1.0f;
        matrix.m[1][1] = 1.0f;
        matrix.m[2][2] = 1.0f;
        matrix.m[3][3] = 1.0f;
        matrix.m[3][0] = x;
        matrix.m[3][1] = y;
        matrix.m[3][2] = z;

        return matrix;
    }

    matrix4x4 Matrix_MakeProjection(float fFovDegrees, float fAspectRatio, float fNear, float fFar) {
        float fFovRad = 1.0f / tanf(fFovDegrees * 0.5f / 180.0f * 3.14159f);

        matrix4x4 matrix;

        matrix.m[0][0] = fAspectRatio * fFovRad;
        matrix.m[1][1] = fFovRad;
        matrix.m[2][2] = fFar / (fFar - fNear);
        matrix.m[3][2] = (-fFar * fNear) / (fFar - fNear);
        matrix.m[2][3] = 1.0f;
        matrix.m[3][3] = 0.0f;

        return matrix;
    }

    matrix4x4 Matrix_MultiplyMatrix(matrix4x4 &m1, matrix4x4 &m2) {
        matrix4x4 matrix;

        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                matrix.m[r][c] = m1.m[r][0] * m2.m[0][c] + m1.m[r][1] * m2.m[1][c] + m1.m[r][2] * m2.m[2][c] + m1.m[r][3] * m2.m[3][c];
 
        return matrix;
    }

    matrix4x4 Matrix_PointAt(vector3d &pos, vector3d &target, vector3d &up) {
        // Calculate new forward direction
        vector3d newForward = Vector_Sub(target, pos);
        newForward = Vector_Normalise(newForward);

        // Calculate new Up direction
        vector3d a = Vector_Mul(newForward, Vector_DotProduct(up, newForward));
        vector3d newUp = Vector_Sub(up, a);
        newUp = Vector_Normalise(newUp);

        // New Right direction is easy, its just cross product
        vector3d newRight = Vector_CrossProduct(newUp, newForward);

        // Construct Dimensioning and Translation Matrix	
        matrix4x4 matrix;

        matrix.m[0][0] = newRight.x;	matrix.m[0][1] = newRight.y;	matrix.m[0][2] = newRight.z;	matrix.m[0][3] = 0.0f;
        matrix.m[1][0] = newUp.x;		matrix.m[1][1] = newUp.y;		matrix.m[1][2] = newUp.z;		matrix.m[1][3] = 0.0f;
        matrix.m[2][0] = newForward.x;	matrix.m[2][1] = newForward.y;	matrix.m[2][2] = newForward.z;	matrix.m[2][3] = 0.0f;
        matrix.m[3][0] = pos.x;			matrix.m[3][1] = pos.y;			matrix.m[3][2] = pos.z;			matrix.m[3][3] = 1.0f;

        return matrix;
    }

    matrix4x4 Matrix_QuickInverse(matrix4x4& m) /* Only for Rotation/Translation Matrices */ {
        matrix4x4 matrix;

        matrix.m[0][0] = m.m[0][0]; matrix.m[0][1] = m.m[1][0]; matrix.m[0][2] = m.m[2][0]; matrix.m[0][3] = 0.0f;
        matrix.m[1][0] = m.m[0][1]; matrix.m[1][1] = m.m[1][1]; matrix.m[1][2] = m.m[2][1]; matrix.m[1][3] = 0.0f;
        matrix.m[2][0] = m.m[0][2]; matrix.m[2][1] = m.m[1][2]; matrix.m[2][2] = m.m[2][2]; matrix.m[2][3] = 0.0f;
        matrix.m[3][0] = -(m.m[3][0] * matrix.m[0][0] + m.m[3][1] * matrix.m[1][0] + m.m[3][2] * matrix.m[2][0]);
        matrix.m[3][1] = -(m.m[3][0] * matrix.m[0][1] + m.m[3][1] * matrix.m[1][1] + m.m[3][2] * matrix.m[2][1]);
        matrix.m[3][2] = -(m.m[3][0] * matrix.m[0][2] + m.m[3][1] * matrix.m[1][2] + m.m[3][2] * matrix.m[2][2]);
        matrix.m[3][3] = 1.0f;

        return matrix;
    }

    vector3d Vector_Add(vector3d& v1, vector3d& v2) {
        return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
    }

    vector3d Vector_Sub(vector3d& v1, vector3d& v2) {
        return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
    }

    vector3d Vector_Mul(vector3d& v1, float k) {
        return { v1.x * k, v1.y * k, v1.z * k };
    }

    vector3d Vector_Div(vector3d& v1, float k) {
        return { v1.x / k, v1.y / k, v1.z / k };
    }

    float Vector_DotProduct(vector3d& v1, vector3d& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    float Vector_Length(vector3d& v) {
        return sqrtf(Vector_DotProduct(v, v));
    }

    vector3d Vector_Normalise(vector3d& v) {
        float l = Vector_Length(v);
        return { v.x / l, v.y / l, v.z / l };
    }

    vector3d Vector_CrossProduct(vector3d& v1, vector3d& v2) {
        vector3d v;

        v.x = v1.y * v2.z - v1.z * v2.y;
        v.y = v1.z * v2.x - v1.x * v2.z;
        v.z = v1.x * v2.y - v1.y * v2.x;

        return v;
    }

    vector3d Vector_IntersectPlane(vector3d& plane_p, vector3d& plane_n, vector3d& lineStart, vector3d& lineEnd) {
        plane_n = Vector_Normalise(plane_n);

        float plane_d = -Vector_DotProduct(plane_n, plane_p);
        float ad = Vector_DotProduct(lineStart, plane_n);
        float bd = Vector_DotProduct(lineEnd, plane_n);
        float t = (-plane_d - ad) / (bd - ad);

        vector3d lineStartToEnd = Vector_Sub(lineEnd, lineStart);
        vector3d lineToIntersect = Vector_Mul(lineStartToEnd, t);

        return Vector_Add(lineStart, lineToIntersect);
    }

    int Triangle_ClipAgainstPlane(vector3d plane_p, vector3d plane_n, triangle& in_tri, triangle& out_tri1, triangle& out_tri2) {
        // Make sure plane normal is indeed normal
        plane_n = Vector_Normalise(plane_n);

        // Return signed shortest distance from point to plane, plane normal must be normalised
        auto dist = [&](vector3d& p) {
            vector3d n = Vector_Normalise(p);
            return (plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - Vector_DotProduct(plane_n, plane_p));
        };

        // Create two temporary storage arrays to classify points either side of plane
        // If distance sign is positive, point lies on "inside" of plane
        vector3d* inside_points[3];  int nInsidePointCount = 0;
        vector3d* outside_points[3]; int nOutsidePointCount = 0;

        // Get signed distance of each point in triangle to plane
        float d0 = dist(in_tri.p[0]);
        float d1 = dist(in_tri.p[1]);
        float d2 = dist(in_tri.p[2]);

        if (d0 >= 0) { inside_points[nInsidePointCount++] = &in_tri.p[0]; }
        else { outside_points[nOutsidePointCount++] = &in_tri.p[0]; }
        if (d1 >= 0) { inside_points[nInsidePointCount++] = &in_tri.p[1]; }
        else { outside_points[nOutsidePointCount++] = &in_tri.p[1]; }
        if (d2 >= 0) { inside_points[nInsidePointCount++] = &in_tri.p[2]; }
        else { outside_points[nOutsidePointCount++] = &in_tri.p[2]; }

        // Now classify triangle points, and break the input triangle into 
        // smaller output triangles if required. There are four possible
        // outcomes...

        if (nInsidePointCount == 0) {
            // All points lie on the outside of plane, so clip whole triangle
            // It ceases to exist

            return 0; // No returned triangles are valid
        }

        if (nInsidePointCount == 3) {
            // All points lie on the inside of plane, so do nothing
            // and allow the triangle to simply pass through
            out_tri1 = in_tri;

            return 1; // Just the one returned original triangle is valid
        }

        if (nInsidePointCount == 1 && nOutsidePointCount == 2) {
            // Triangle should be clipped. As two points lie outside
            // the plane, the triangle simply becomes a smaller triangle

            // Copy appearance info to new triangle
            out_tri1.color = olc::GREEN;

            // The inside point is valid, so keep that...
            out_tri1.p[0] = *inside_points[0];

            // but the two new points are at the locations where the 
            // original sides of the triangle (lines) intersect with the plane
            out_tri1.p[1] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[0]);
            out_tri1.p[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[1]);

            return 1; // Return the newly formed single triangle
        }

        if (nInsidePointCount == 2 && nOutsidePointCount == 1) {
            // Triangle should be clipped. As two points lie inside the plane,
            // the clipped triangle becomes a "quad". Fortunately, we can
            // represent a quad with two new triangles

            // Copy appearance info to new triangles
            out_tri1.color = olc::RED;
            out_tri2.color = olc::BLUE;

            // The first triangle consists of the two inside points and a new
            // point determined by the location where one side of the triangle
            // intersects with the plane
            out_tri1.p[0] = *inside_points[0];
            out_tri1.p[1] = *inside_points[1];
            out_tri1.p[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[0]);

            // The second triangle is composed of one of he inside points, a
            // new point determined by the intersection of the other side of the 
            // triangle and the plane, and the newly created point above
            out_tri2.p[0] = *inside_points[1];
            out_tri2.p[1] = out_tri1.p[2];
            out_tri2.p[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[1], *outside_points[0]);

            return 2; // Return two newly formed triangles which form a quad
        }
    }

public:
    bool OnUserCreate() override {
        cube.triangles = {
            // South
            {0.0f, 0.0f, 0.0f, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,  1.0f, 1.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f, 1.0f,  1.0f, 1.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f},
            // East
            {1.0f, 0.0f, 0.0f, 1.0f,  1.0f, 1.0f, 0.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f},
            {1.0f, 0.0f, 0.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 0.0f, 1.0f, 1.0f},
            // North
            {1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f},
            {1.0f, 0.0f, 1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f,  0.0f, 0.0f, 1.0f, 1.0f},
            // West
            {0.0f, 0.0f, 1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 0.0f, 1.0f},
            // Top
            {0.0f, 1.0f, 0.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f},
            {0.0f, 1.0f, 0.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 0.0f, 1.0f},
            // Bottom
            {1.0f, 0.0f, 1.0f, 1.0f,  0.0f, 0.0f, 1.0f, 1.0f,  0.0f, 0.0f, 0.0f, 1.0f},
            {1.0f, 0.0f, 1.0f, 1.0f,  0.0f, 0.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f},
        };

        projection = Matrix_MakeProjection(90.0f, (float)ScreenHeight() / (float)ScreenWidth(), 0.1f, 1000.0f);

        return true;
    }

public:
    bool OnUserUpdate(float elapsedTime) override {
        if (GetKey(olc::UP).bHeld)
            camera.y += 8.0f * elapsedTime;	// Travel Upwards

        if (GetKey(olc::DOWN).bHeld)
            camera.y -= 8.0f * elapsedTime;	// Travel Downwards


        // Dont use these two in FPS mode, it is confusing :P
        if (GetKey(olc::LEFT).bHeld)
            camera.x -= 8.0f * elapsedTime;	// Travel Along X-Axis

        if (GetKey(olc::RIGHT).bHeld)
            camera.x += 8.0f * elapsedTime;	// Travel Along X-Axis

        vector3d forward = Vector_Mul(lookDirection, 8.0f * elapsedTime);

        // Standard FPS Control scheme, but turn instead of strafe
        if (GetKey(olc::W).bHeld)
            camera = Vector_Add(camera, forward);

        if (GetKey(olc::S).bHeld)
            camera = Vector_Sub(camera, forward);

        if (GetKey(olc::A).bHeld)
            yaw -= 2.0f * elapsedTime;

        if (GetKey(olc::D).bHeld)
            yaw += 2.0f * elapsedTime;

        matrix4x4 rotationZ, rotationX, translation, world, view;

        theta += 1.0f * elapsedTime;

        rotationZ   = Matrix_MakeRotationZ(theta * 0.5f);
        rotationX   = Matrix_MakeRotationX(theta);
        translation = Matrix_MakeTranslation(0.0f, 0.0f, 5.0f);

        world = Matrix_MakeIdentity();	// Form World Matrix
        world = Matrix_MultiplyMatrix(world, rotationZ); // Transform by rotation
        world = Matrix_MultiplyMatrix(world, rotationX); // Transform by rotation
        world = Matrix_MultiplyMatrix(world, translation); // Transform by translation

        // Create "Point At" Matrix for camera
        vector3d vUp = { 0,1,0 };
        vector3d target = { 0,0,1 };
        matrix4x4 cameraRotation = Matrix_MakeRotationY(yaw);

        lookDirection = Matrix_MultiplyVector(cameraRotation, target);
        target = Vector_Add(camera, lookDirection);

        matrix4x4 matCamera = Matrix_PointAt(camera, target, vUp);

        // Make view matrix from camera
        view = Matrix_QuickInverse(matCamera);

        // Store triagles for rastering later
        vector<triangle> toRaster;

        for (auto t : cube.triangles) {
            triangle projected, transformed, viewable;

            // World Matrix Transform
            transformed.p[0] = Matrix_MultiplyVector(world, t.p[0]);
            transformed.p[1] = Matrix_MultiplyVector(world, t.p[1]);
            transformed.p[2] = Matrix_MultiplyVector(world, t.p[2]);

            vector3d normal, lineA, lineB;

            lineA = Vector_Sub(transformed.p[1], transformed.p[0]);
            lineB = Vector_Sub(transformed.p[2], transformed.p[0]);

            normal = Vector_CrossProduct(lineA, lineB);
            normal = Vector_Normalise(normal);

            // Get Ray from triangle to camera
            vector3d ray = Vector_Sub(transformed.p[0], camera);

            if (Vector_DotProduct(normal, ray) < 0.0f) {
                vector3d directionalLight = {0.0f, 0.0f, -1.0f};

                directionalLight = Vector_Normalise(directionalLight);

                // How similar is normal to light direction
                float dp = max(0.0001f, Vector_DotProduct(directionalLight, normal));

                // Get value into 1 - 256 range so it can be used to create the color below
                float intensity = dp * 255.0f;

                transformed.color = olc::Pixel(intensity, intensity, intensity);

                /// Convert World Space -> View Space
                viewable.p[0] = Matrix_MultiplyVector(view, transformed.p[0]);
                viewable.p[1] = Matrix_MultiplyVector(view, transformed.p[1]);
                viewable.p[2] = Matrix_MultiplyVector(view, transformed.p[2]);

                // Clip Viewed Triangle against near plane, this could form two additional
                // additional triangles. 
                int clippedTriangles = 0;

                triangle clipped[2];

                clippedTriangles = Triangle_ClipAgainstPlane({ 0.0f, 0.0f, 0.1f }, { 0.0f, 0.0f, 1.0f }, viewable, clipped[0], clipped[1]);

                // We may end up with multiple triangles form the clip, so project as
                // required
                for (int n = 0; n < clippedTriangles; n++) {
                    // Project triangles from 3D --> 2D
                    projected.p[0] = Matrix_MultiplyVector(projection, clipped[n].p[0]);
                    projected.p[1] = Matrix_MultiplyVector(projection, clipped[n].p[1]);
                    projected.p[2] = Matrix_MultiplyVector(projection, clipped[n].p[2]);

                    projected.color = transformed.color;

                    // Scale into view, we moved the normalising into cartesian space
                    // out of the matrix.vector function from the previous videos, so
                    // do this manually
                    projected.p[0] = Vector_Div(projected.p[0], projected.p[0].w);
                    projected.p[1] = Vector_Div(projected.p[1], projected.p[1].w);
                    projected.p[2] = Vector_Div(projected.p[2], projected.p[2].w);

                    // X/Y are inverted so put them back
                    projected.p[0].x *= -1.0f;
                    projected.p[1].x *= -1.0f;
                    projected.p[2].x *= -1.0f;
                    projected.p[0].y *= -1.0f;
                    projected.p[1].y *= -1.0f;
                    projected.p[2].y *= -1.0f;

                    // Offset verts into visible normalised space
                    vector3d vOffsetView = { 1,1,0 };

                    projected.p[0] = Vector_Add(projected.p[0], vOffsetView);
                    projected.p[1] = Vector_Add(projected.p[1], vOffsetView);
                    projected.p[2] = Vector_Add(projected.p[2], vOffsetView);
                    projected.p[0].x *= 0.5f * (float)ScreenWidth();
                    projected.p[0].y *= 0.5f * (float)ScreenHeight();
                    projected.p[1].x *= 0.5f * (float)ScreenWidth();
                    projected.p[1].y *= 0.5f * (float)ScreenHeight();
                    projected.p[2].x *= 0.5f * (float)ScreenWidth();
                    projected.p[2].y *= 0.5f * (float)ScreenHeight();

                    // Store triangle for sorting
                    toRaster.push_back(projected);
                }
            }
        }

        // Sort triangles from back to front
        sort(toRaster.begin(), toRaster.end(), [](triangle& t1, triangle& t2) {
            float z1 = (t1.p[0].z + t1.p[1].z + t1.p[2].z) / 3.0f;
            float z2 = (t2.p[0].z + t2.p[1].z + t2.p[2].z) / 3.0f;

            return z1 > z2;
        });
        
        FillRect(0, 0, ScreenWidth(), ScreenHeight(), olc::BLACK);

        // Loop through all transformed, viewed, projected, and sorted triangles
        for (auto& r : toRaster) {
            // Clip triangles against all four screen edges, this could yield
            // a bunch of triangles, so create a queue that we traverse to 
            //  ensure we only test new triangles generated against planes
            triangle clipped[2];
            list<triangle> listTriangles;

            // Add initial triangle
            listTriangles.push_back(r);
            int nNewTriangles = 1;

            for (int p = 0; p < 4; p++) {
                int nTrisToAdd = 0;
                while (nNewTriangles > 0) {
                    // Take triangle from front of queue
                    triangle test = listTriangles.front();
                    listTriangles.pop_front();
                    nNewTriangles--;

                    // Clip it against a plane. We only need to test each 
                    // subsequent plane, against subsequent new triangles
                    // as all triangles after a plane clip are guaranteed
                    // to lie on the inside of the plane. I like how this
                    // comment is almost completely and utterly justified
                    switch (p)  {
                        case 0:	nTrisToAdd = Triangle_ClipAgainstPlane({ 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                        case 1:	nTrisToAdd = Triangle_ClipAgainstPlane({ 0.0f, (float)ScreenHeight() - 1, 0.0f }, { 0.0f, -1.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                        case 2:	nTrisToAdd = Triangle_ClipAgainstPlane({ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                        case 3:	nTrisToAdd = Triangle_ClipAgainstPlane({ (float)ScreenWidth() - 1, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                    }

                    // Clipping may yield a variable number of triangles, so
                    // add these new ones to the back of the queue for subsequent
                    // clipping against next planes
                    for (int w = 0; w < nTrisToAdd; w++)
                        listTriangles.push_back(clipped[w]);
                }
                nNewTriangles = listTriangles.size();
            }


            // Draw the transformed, viewed, clipped, projected, sorted, clipped triangles
            for (auto& t : listTriangles) {
                FillTriangle(t.p[0].x, t.p[0].y, t.p[1].x, t.p[1].y, t.p[2].x, t.p[2].y, t.color);
                // DrawTriangle(t.p[0].x, t.p[0].y, t.p[1].x, t.p[1].y, t.p[2].x, t.p[2].y, olc::BLACK);
            }
        }

        return true;
    }
};

int main() {
    olcConsoleEngine3D demo;

    if (demo.Construct(256, 240, 4, 4))
        demo.Start();

    return 0;
}
