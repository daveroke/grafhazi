//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2018-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kovács Dávid	
// Neptun : UT55G0
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#define _USE_MATH_DEFINES		// Van M_PI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) { printf("%s!\n", message); getErrorInfo(shader); }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) { printf("Failed to link shader program!\n"); getErrorInfo(program); }
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

struct vec3 {
	float x, y, z;

	vec3(float x0, float y0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) { // vektor, szín, pont + vektor
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) { // vektor, szín, pont - pont
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	float Length() { return sqrtf(x * x + y * y + z * z); }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);
}


// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) const {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};

// 3D point in homogeneous coordinates
struct vec4 {
	float x, y, z, w;
	vec4(float _x = 0, float _y = 0, float _z = 0, float _w = 1) { x = _x; y = _y; z = _z; w = _w; }

	vec4 operator*(const mat4& mat) const {
		return vec4(x * mat.m[0][0] + y * mat.m[1][0] + z * mat.m[2][0] + w * mat.m[3][0],
			x * mat.m[0][1] + y * mat.m[1][1] + z * mat.m[2][1] + w * mat.m[3][1],
			x * mat.m[0][2] + y * mat.m[1][2] + z * mat.m[2][2] + w * mat.m[3][2],
			x * mat.m[0][3] + y * mat.m[1][3] + z * mat.m[2][3] + w * mat.m[3][3]);
	}
};

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
	unsigned int vao;	// vertex array object id
	float phi;			// rotation
public:
	Triangle() {
		Animate(0);
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		float vertexCoords[] = { -8, -8,   -6, 10,   8, -2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		float vertexColors[] = { 1, 0, 0,   0, 1, 0,   0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) { phi = t; }

	void Draw() {
		mat4 MVPTransform(0.1 * cos(phi), 0.1 * sin(phi), 0, 0,
			-0.1 * sin(phi), 0.1 * cos(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class Circle {
	unsigned int vao;	// vertex array object id
	float phi;			// rotation
	int leaves;
	float radius;
	float xc;
	float yc;
public:
	Circle() {
		Animate(0);
	}

	void Create(float rad, float cx, float cy, int r, int g, int b, int leaf) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array

		xc = cx;
		yc = cy;
		radius = rad;
		leaves = leaf;
		float x[50];
		float y[50];

		for (int i = 0; i < 50; i++) {			
			x[i] = xc + rad*cosf(i*(2*M_PI/50));
			y[i] = yc + rad*sinf(i*(2*M_PI/50));
		}

		float vertexCoords[100];	// vertex data on the CPU
		int t = 0;
		for (int i = 0; i < 100; i++) {			
			vertexCoords[t] = x[i];
			vertexCoords[t + 1] = y[i];
			t = t + 2;
		}

		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),   // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);
		
			   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array

		float vertexColors[150];	// vertex data on the CPU

		for (int i = 0; i < 150; i=i+3) {
			vertexColors[i] = r;
			vertexColors[i + 1] = g;
			vertexColors[i + 2] = b;
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}
	
	void Animate(float t) { phi = t; }

	void Draw() {
		mat4 MVPTransform(  1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 1);

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 50);	// draw a single triangle with vertices defined in vao
	}
	int getLeaves() {
		return leaves;
	}

	float getRad() {
		return radius;
	}

	float getCx() {
		return xc;
	}

	float getCy() {
		return yc;
	}
};

class Leaves {
	unsigned int vao;	// vertex array object id
	float phi;			// rotation
	float radius;
	float x;
	float y;
public:
	Leaves() {
		Animate(0);
	}

	void Create(float cx, float cy, float yo, float rad, float fi, int r, int g, int b) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		x = cx;
		y = cy;
		phi = fi;
		float c[] = { cx, cy };
		float x[50];
		float y[50];

		for (int i = 0; i < 50; i++) {
			x[i] = c[0] + rad * cosf(i*(2 * M_PI / 50));
			y[i] = c[1] + rad * sinf(i*(2 * M_PI / 50)) / yo;
		}

		float vertexCoords[100];	// vertex data on the CPU
		int t = 0;
		for (int i = 0; i < 100; i++) {
			vertexCoords[t] = x[i];
			vertexCoords[t + 1] = y[i];
			t = t + 2;
		}

		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),   // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);

		// copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array

		float vertexColors[150];	// vertex data on the CPU

		for (int i = 0; i < 150; i = i + 3) {
			vertexColors[i] = r;
			vertexColors[i + 1] = g;
			vertexColors[i + 2] = b;
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) { phi = t; }

	void Draw() {
		mat4 MVPTransform(	cos(phi), sin(phi), 0, 0,
							-sin(phi), cos(phi), 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 1);

		mat4 pushMatrix(	1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 0,
						   -x, -y, 0, 1);

		mat4 pushBackMatrix(	1, 0, 0, 0,
								0, 1, 0, 0,
								0, 0, 1, 0,
								x, y, 0, 1);

		mat4 multi = pushMatrix * MVPTransform * pushBackMatrix;


		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, multi); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 50);	// draw a single triangle with vertices defined in vao
	}
};

class Butterfly {
	unsigned int vao;	// vertex array object id
	float phi;			// rotation
	float radius;
	float x;
	float y;
public:
	Butterfly() {
		Animate(0);
	}

	void Create(float cx, float cy, float yo, float rad, int r, int g, int b) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		x = cx;
		y = cy;
		float c[] = { cx, cy };
		float x[50];
		float y[50];

		for (int i = 0; i < 50; i++) {
			x[i] = c[0] + rad * cosf(i*(2 * M_PI / 50));
			y[i] = c[1] + rad * sinf(i*(2 * M_PI / 50)) / yo;
		}

		float vertexCoords[100];	// vertex data on the CPU
		int t = 0;
		for (int i = 0; i < 100; i++) {
			vertexCoords[t] = x[i];
			vertexCoords[t + 1] = y[i];
			t = t + 2;
		}

		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),   // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);

		// copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array

		float vertexColors[150];	// vertex data on the CPU

		for (int i = 0; i < 150; i = i + 3) {
			vertexColors[i] = r;
			vertexColors[i + 1] = g;
			vertexColors[i + 2] = b;
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) { phi = t; }

	void Draw() {
		mat4 initMatrix(	1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 1);

		/*mat4 MVPTransform(cos(phi), sin(phi), 0, 0,
			-sin(phi), cos(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		mat4 pushMatrix(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-x, -y, 0, 1);

		mat4 pushBackMatrix(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			x, y, 0, 1);

		mat4 multi = pushMatrix * MVPTransform * pushBackMatrix;*/


		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, initMatrix); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 50);	// draw a single triangle with vertices defined in vao
	}
};

class Wing {
	unsigned int vao;	// vertex array object id
	float phi;			// rotation
	float x;
	float y;
public:
	Wing() {
		Animate(0);
	}

	void Create(float cx, float cy) {

		x = cx;
		y = cy;

		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		float vertexCoords[] = { x, y, 0.1, -0.1, -0.1, -0.1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		float vertexColors[] = { 0, 1, 0,   0, 1, 0,   0, 1, 0 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) { phi = t; }

	void Draw(bool mir) {
		mat4 MVPTransform(	1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 1);

		mat4 pushOrigo(		1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 0,
							-x, -y, 0, 1);

		mat4 rotateZ(		1, 0, 0, 0,
							0, cos(phi), sin(phi), 0,
							0, -sin(phi), cos(phi), 0,
							0, 0, 0, 1);

		mat4 mirror(		-1, 0, 0, 0,
							0, -1, 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 1);

		mat4 pushBack(	1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						x, y, 0, 1);

		mat4 fin;
		if (mir) {
			fin = pushOrigo * rotateZ * mirror * MVPTransform * pushBack;
		}
		else {
			fin = pushOrigo * rotateZ * MVPTransform * pushBack;
		}

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, fin); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

// The virtual world
//Flower middles
Circle middle;
Circle middle2;
Circle middle3;
Circle middle4;
Circle middle5;
Leaves* leaves1;
Leaves* leaves2;
Leaves* leaves3;
Leaves* leaves4;
Leaves* leaves5;
Butterfly butterfly;
Wing wing;
Wing wing2;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	//first flower init
	middle.Create(0.1, 0, 0, 1, 1, 0, 3);
	float cx1[3];
	float cy1[3];
	float phi1[3];

	for (int i = 0; i < 3; i++) {
		cx1[i] = middle.getCx() + 0.1 * cosf(i*(2 * M_PI / 3));
		cy1[i] = middle.getCy() + 0.1 * sinf(i*(2 * M_PI / 3));
		phi1[i] = i * (2 * M_PI / 3);
	}
	leaves1 = new Leaves[3];
	for (int i = 0; i < 3; i++) {
		leaves1[i].Create(cx1[i], cy1[i], 1, 0.1, phi1[i], 0, 0, 1);
	}
	//second flower init
	middle2.Create(0.1, 0.5, 0.5, 1, 1, 0, 5);
	float cx2[5];
	float cy2[5];
	float phi2[5];

	for (int i = 0; i < 5; i++) {
		cx2[i] = middle2.getCx() + 0.1 * cosf(i*(2 * M_PI / 5));
		cy2[i] = middle2.getCy() + 0.1 * sinf(i*(2 * M_PI / 5));
		phi2[i] = i * (2 * M_PI / 5);
	}
	leaves2 = new Leaves[5];
	for (int i = 0; i < 5; i++) {
		leaves2[i].Create(cx2[i], cy2[i], 2, 0.1, phi2[i], 0, 0, 1);
	}
	//third flower init
	middle3.Create(0.1, 0.5, -0.5, 1, 1, 0, 8);
	float cx3[8];
	float cy3[8];
	float phi3[8];
	for (int i = 0; i < 8; i++) {
		cx3[i] = middle3.getCx() + 0.1 * cosf(i*(2 * M_PI / 8));
		cy3[i] = middle3.getCy() + 0.1 * sinf(i*(2 * M_PI / 8));
		phi3[i] = i * (2 * M_PI / 8);
	}
	leaves3 = new Leaves[8];
	for (int i = 0; i < 8; i++) {
		leaves3[i].Create(cx3[i], cy3[i], 3, 0.1, phi3[i], 0, 0, 1);
	}
	//fourth flower init
	middle4.Create(0.1, -0.5, 0.5, 1, 1, 0, 13);
	float cx4[13];
	float cy4[13];
	float phi4[13];

	for (int i = 0; i < 13; i++) {
		cx4[i] = middle4.getCx() + 0.1 * cosf(i*(2 * M_PI / 13));
		cy4[i] = middle4.getCy() + 0.1 * sinf(i*(2 * M_PI / 13));
		phi4[i] = i * (2 * M_PI / 13);
	}
	leaves4 = new Leaves[13];
	for (int i = 0; i < 13; i++) {
		leaves4[i].Create(cx4[i], cy4[i], 4, 0.1, phi4[i], 0, 0, 1);
	}
	//fifth flower init
	middle5.Create(0.1, -0.5, -0.5, 1, 1, 0, 21);
	float cx5[21];
	float cy5[21];
	float phi5[21];

	for (int i = 0; i < 21; i++) {
		cx5[i] = middle5.getCx() + 0.1 * cosf(i*(2 * M_PI / 21));
		cy5[i] = middle5.getCy() + 0.1 * sinf(i*(2 * M_PI / 21));
		phi5[i] = i * (2 * M_PI / 21);
	}
	leaves5 = new Leaves[21];
	for (int i = 0; i < 21; i++) {
		leaves5[i].Create(cx5[i], cy5[i], 5, 0.1, phi5[i], 0, 0, 1);
	}

	//butterfly init
	butterfly.Create(0, 0, 2, 0.08, 1, 0, 1);
	wing.Create(0, 0);
	wing2.Create(0, 0);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
	delete leaves1;
	delete leaves2;
	delete leaves3;
	delete leaves4;
	delete leaves5;
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0.2, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	//first flower draw
	for (int i = 0; i < middle.getLeaves(); i++) {
		leaves1[i].Draw();
	}
	middle.Draw();
	//second flower draw
	for (int i = 0; i < middle2.getLeaves(); i++) {
		leaves2[i].Draw();
	}
	middle2.Draw();
	//third flower draw
	for (int i = 0; i < middle3.getLeaves(); i++) {
		leaves3[i].Draw();
	}
	middle3.Draw();
	//fourth flower draw
	for (int i = 0; i < middle4.getLeaves(); i++) {
		leaves4[i].Draw();
	}
	middle4.Draw();
	//fifth flower draw
	for (int i = 0; i < middle5.getLeaves(); i++) {
		leaves5[i].Draw();
	}
	middle5.Draw();
	//butterfly draw
	wing.Draw(true);
	wing2.Draw(false);
	butterfly.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	wing.Animate(sec);					// animate the triangle object
	wing2.Animate(sec);
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}