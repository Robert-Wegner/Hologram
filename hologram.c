#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
//nvcc -c hologram_cuda.cu
//gcc -lm -lcuda -L/opt/cuda/lib64/ -lcudart holo_cuda_launch.c hologram_cuda.o -o hologram_cuda $(pkg-config --libs allegro-5 allegro_primitives-5)
//gcc -std=c99 -lm hologram.c -o hologram $(pkg-config --libs allegro-5 allegro_primitives-5)





#define W0 24
#define W1 32
#define CAMERAS 16
#define SCALE 0.4
#define DISTANCE_TO_SCREEN 0.05
#define RADIUS SCALE / cos(M_PI / 6.0)
#define HEIGHT sqrt(pow(RADIUS, 2) - pow(SCALE, 2))
#define SLOPE SCALE / sqrt(3.0)
#define QUOTIENT tan(1/6 * M_PI) * 0.5 * RADIUS / CAMERAS


typedef struct {
    float density;
    float density_change;
    int resolution[2];
    float x;
    float y;
    float z;
    int color[3];
    int grid;
    int FPS;
    float* P;
    int* Pc;
    int lenP;
    int maxP;
    int saving_mode;
    float offset;
    float zoom_change;
    int rmb_mode;
    float line_density;
    float prevx, prevy, prevz;
    

} PROG_STATE;

void project(float*, float*, float*, int);
int** getTriDict(int);
int** getInvTriDict(int);
void posToHexagonIndex(float*, int*, int);
void indexToHexagonPos(int, int, float*);
void indexToTrianglePos(int, int, float*);
float* generatePoints(int, int);
int* generateColors(int, int);
void handleMouseMove(ALLEGRO_MOUSE_EVENT, PROG_STATE*);
void handleLMBDown(ALLEGRO_MOUSE_EVENT, PROG_STATE*);
void updateCursor(PROG_STATE*);
void handleColorChange(PROG_STATE*, int, int, int);
void handleSwitchSavingMode(PROG_STATE*);
void handleSavePoints(PROG_STATE*, char);
void handleKeyboard(PROG_STATE*, int);
void handleSwitchGrid(PROG_STATE*);
void handleChangeDensity(PROG_STATE*, int);
void handleChangeZoom(PROG_STATE*, int);
void handleRMBDown(ALLEGRO_MOUSE_EVENT, PROG_STATE*);
void addPoint(PROG_STATE* state, float, float, float, int, int, int);


void printFloatArray(float* array, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f, ", array[i]);
    }
}

PROG_STATE* initializeState() {
    PROG_STATE* state = malloc(sizeof(PROG_STATE));
    state->x = 0;
    state->y = 0;
    state->z = 5;
    state->color[0] = 255;
    state->color[1] = 0;
    state->color[2] = 255;
    state->resolution[0] = 2700;
    state->resolution[1] = 1700;
    state->grid = 1;
    state->FPS = 60;
    state->density = 106;
    state->density_change = 1.0025;
    state->maxP = 300000;
    state->lenP = 1;
    state->saving_mode = 0;
    state->zoom_change = 1.1;
    state->rmb_mode = 0;
    state->line_density = 0.1;
    return(state);
}

int tDiv(float a, float b) {
    return((int)(floor(a/b)));
}

float tMod(float a, float b) {
    return(a - tDiv(a, b)*b);
}

void project(float* P, float* v, float* target, int n) {
    target[2*n] = P[3*n] - v[0] / v[2] * P[3*n+2];
    target[2*n+1] = P[3*n+1] - v[1] / v[2] * P[3*n+2];
}

int getLenV(int c) {
    int max = 0;
    for (int i = 0; i < 4 * c; i++) {
        if (i < 2 * c) {
            for (int j = 0; j < c + tDiv(i+1, 2); j++) {
                max++;
            }
        }
        else {
            for (int j = 0; j < c + tDiv(4*c - i, 2); j++) {
                max++;
            }
        }
    }
    return(max);
}

int** getTriDict(int c) {


    int** triDict = (int**) malloc(getLenV(c) * sizeof(int*));
    int k = 0;
    for (int i = 0; i < 4 * c; i++) {
        if (i < 2 * c) {
            for (int j = 0; j < c + tDiv(i+1, 2); j++) {

                triDict[k] = (int*) malloc(2 * sizeof(int));
                triDict[k][0] = i;
                triDict[k][1] = j;

                k++;
            }
        }
        else {
            for (int j = 0; j < c + tDiv(4*c - i, 2); j++) {

                triDict[k] = (int*) malloc(2 * sizeof(int));
                triDict[k][0] = i;
                triDict[k][1] = j;

                k++;
            }
        }
    }
    return(triDict);
}

int** getTriDictInv(int c) {

    int** triDictInv = (int**) malloc(4 * c * sizeof(int*));
    int k = 0;
    for (int i = 0; i < 4 * c; i++) {
        if (i < 2 * c) {
            triDictInv[i] = (int*) malloc((c + tDiv(i+1, 2)) * sizeof(int));
            for (int j = 0; j < c + tDiv(i+1, 2); j++) {

                triDictInv[i][j] = k;
                k++;
            }
        }
        else {
            triDictInv[i] = (int*) malloc((c + tDiv(4*c - i, 2)) * sizeof(int));
            for (int j = 0; j < c + tDiv(4*c - i, 2); j++) {

                triDictInv[i][j] = k;
                k++;
            }
        }
    }
    return(triDictInv);
}


int** allocHologram(int lenV, int N, int M) {
    int** array = (int**) malloc(lenV * sizeof(int*));
    for (int i = 0; i < lenV; i++) {
        array[i] = malloc(N * M * 3 * sizeof(int));
        for (int k = 0; k < N * M * 3; k++) {
            array[i][k] = 0;
        }
    }
    return(array);
}

float* allocDistances(int N, int M) {
    float* array = malloc(N * M * sizeof(float*));
    return(array);
}

ALLEGRO_VERTEX* allocVertices(int N, int M, int lenV) {
    return((ALLEGRO_VERTEX*) malloc(N * M * 3 * lenV * sizeof(ALLEGRO_VERTEX)));
}

void setDistancesToInf(float* distances, int N, int M) {
    for (int i = 0; i < N * M; i++) {
        distances[i] = INFINITY;
    }
}

void setXToBlack(int** X, int N, int M, int lenV) {
    for (int i = 0; i < lenV; i++) {
        for (int k = 0; k < 3 * N * M; k++) {
            X[i][k] = 0;
        }
    }
}

float** generateViews(int lenV, int** triDict) {
    float** V = (float**) malloc(sizeof(float*) * lenV);
    float* pos = (float*) malloc(sizeof(float) * 2);
    for (int i = 0; i < lenV; i++) {
        V[i] = (float*) malloc(sizeof(float) * 3);
        indexToTrianglePos(triDict[i][0], triDict[i][1], pos);
        V[i][0] = pos[0];
        V[i][1] = pos[1];
        V[i][2] = SCALE + DISTANCE_TO_SCREEN;
    }
    return(V);
}

void posToHexagonIndex(float* projected, int* target, int n) {
    
    float x = projected[2*n];
    float y = projected[2*n+1];
    float s = SCALE;
    float r = RADIUS;
    float h = HEIGHT;
    float m = SLOPE;
    
    y = y + h;

    int k = tDiv(y, (r + h)) + tDiv(W0, 2);
    x = x + s + s * tMod(k, 2);
    
    int l = tDiv(x, (2 * s)) + tDiv(W1, 2);

    float dx = tMod(x, (2 * s)) - s;
    float dy = tMod(y, (r + h)) - h;
    
    if (dx < 0 && dy > m * dx + r) {
        target[2*n] = k + 1;
        target[2*n+1] = l - (int)(tMod(k, 2));
    }
    else if (dx >= 0 && dy >= -m * dx + r) {
        target[2*n] = k + 1;
        target[2*n+1] = 1 + (int)(l - tMod(k, 2));
    }
    else {
        target[2*n] = k;
        target[2*n+1] = l;
    }
}

void indexToHexagonPos(int k, int l, float* target) {        

    float s = SCALE;
    float r = RADIUS;

    float lmod = l - (W1) / 2.0 - tMod(k, 2) / 2.0;
    float kmod = k - (W0) / 2.0;

    target[0] = 2 * s * lmod;
    target[1] = 3.0/2.0 * r * kmod;

}

void indexToTrianglePos(int i, int j, float* target) {          

    int c = CAMERAS;
    float s = SCALE;
    float r = RADIUS;
    float q = QUOTIENT;
    int sign;
    int imod;
    float jmod;
    if (i < 2 * c) {
        jmod = (j - (c + tDiv(i+1, 2) - 1) / 2.0);
        imod = i - 2 * c;
    }
    else {
        jmod = (j - (c + tDiv(4*c-i, 2) - 1) / 2.0);    
        imod = i + 1 - 2 * c;
    }
    
    target[1] = jmod * r / c;
    sign = (imod > 0) - (imod < 0);
    
    if (round(tMod(i, 2)) == 0) {
        target[0] = sign*tDiv(sign*imod, 2) * s/c - q * sign;
    }
    else {
        target[0] = sign*tDiv(sign*imod, 2) * s/c + q * sign;
    }
    

}

void processPoint(float* P, int* Pc, float* v, int* target, float* distances, float* projected, int* ind, int n) {
    
    int k, l;
    project(P, v, projected, n);
    posToHexagonIndex(projected, ind, n);
    k = ind[2*n];
    l = ind[2*n+1];

    if (0 <= k && k < W0 && 0 <= l && l < W1) {
        if (P[3*n+2] < distances[k*W1 + l]) {
            distances[k*W1 + l] = P[3*n+2];
            target[k*W1*3 + l*3 + 0] = Pc[3*n+0];
            target[k*W1*3 + l*3 + 1] = Pc[3*n+1];
            target[k*W1*3 + l*3 + 2] = Pc[3*n+2];
        }
    }
}

void processView(float* P, int* Pc, int lenP, float* v, int* target, float* distances, float* projected, int* ind) {
    for (int n = 0; n < lenP; n++) {
        processPoint(P, Pc, v, target, distances, projected, ind, n);
    }

}

void createHologram(float* P, int* Pc, int lenP, float** V, int lenV, int** X, float* distances, float* projected, int* ind) {
    //printf("createHologram \n");

    setXToBlack(X, W0, W1, lenV);
    for (int i = 0; i < lenV; i++) {
        setDistancesToInf(distances, W0, W1);
        processView(P, Pc, lenP, V[i], X[i], distances, projected, ind);
    }
    //printf("here\n");
}

void initializeDisplayVertices( float** V, int lenV, ALLEGRO_VERTEX* vertices,
                                int** triDict, int* resolution, float density, float* triPos, float* hexPos) {
    int w = resolution[0];
    int h = resolution[1];
    int imod = 0;
    int c = CAMERAS;
    float s = SCALE;
    float r = RADIUS;
    float q  = QUOTIENT;
    ALLEGRO_VERTEX_BUFFER* buffer = NULL;
    for (int k = 0; k < W0; k++) { 
        for (int l = 0; l < W1; l++) {

            indexToHexagonPos(k, l, hexPos);

            for (int i = 0; i < lenV; i++) {

                indexToTrianglePos(triDict[i][0], triDict[i][1], triPos);

                triPos[0] += hexPos[0];
                triPos[1] += hexPos[1];

                if (triDict[i][0] % 2 == 0) {
                    vertices[imod].x = (triPos[0] - q)*density+w/2.0;
                    vertices[imod].y = (triPos[1] + 0.5 * r/c)*density+h/2.0;
                    vertices[imod].z = 0;
                    vertices[imod].u = 0;
                    vertices[imod].v = 0;
                    vertices[imod].color = al_map_rgb(0, 0, 0);
                    imod++;                    
                    vertices[imod].x = (triPos[0] - q)*density+w/2.0;
                    vertices[imod].y = (triPos[1] - 0.5 * r/c)*density+h/2.0;
                    vertices[imod].z = 0;
                    vertices[imod].u = 0;
                    vertices[imod].v = 0;
                    vertices[imod].color = al_map_rgb(0, 0, 0);
                    imod++;                    
                    vertices[imod].x = (triPos[0] + s/c - q)*density+w/2.0;
                    vertices[imod].y = (triPos[1])*density+h/2.0;
                    vertices[imod].z = 0;
                    vertices[imod].u = 0;
                    vertices[imod].v = 0;
                    vertices[imod].color = al_map_rgb(0, 0, 0);
                    imod++;
                }                
                else {                    
                    vertices[imod].x = (triPos[0] + q)*density+w/2.0;
                    vertices[imod].y = (triPos[1] + 0.5 * r/c)*density+h/2.0;
                    vertices[imod].z = 0;
                    vertices[imod].u = 0;
                    vertices[imod].v = 0;
                    vertices[imod].color = al_map_rgb(0, 0, 0);
                    imod++;                    
                    vertices[imod].x = (triPos[0] + q)*density+w/2.0;
                    vertices[imod].y = (triPos[1] - 0.5 * r/c)*density+h/2.0;
                    vertices[imod].z = 0;
                    vertices[imod].u = 0;
                    vertices[imod].v = 0;
                    vertices[imod].color = al_map_rgb(0, 0, 0);
                    imod++;                    
                    vertices[imod].x = (triPos[0] - s/c + q)*density+w/2.0;
                    vertices[imod].y = (triPos[1])*density+h/2.0;
                    vertices[imod].z = 0;
                    vertices[imod].u = 0;
                    vertices[imod].v = 0;
                    vertices[imod].color = al_map_rgb(0, 0, 0);
                    imod++;
                }
            }
        }
    }
}

void setDisplayVerticesBlack(int lenV, ALLEGRO_VERTEX* vertices) {
    int imod = 0;

    for (int k = 0; k < W0; k++) { 
        for (int l = 0; l < W1; l++) {
            for (int i = 0; i < lenV; i++) {
                vertices[imod].color = al_map_rgb(0, 0, 0);
                imod++;                    
                vertices[imod].color = al_map_rgb(0, 0, 0);
                imod++;                    
                vertices[imod].color = al_map_rgb(0, 0, 0);
                imod++;
            }
        }
    }
}

void displayHologram(   int** X, float** V, int lenV, ALLEGRO_VERTEX* vertices,
                        int** triDict, int* resolution, float density, bool grid, float* triPos, float* hexPos) {
    //printf("displayHologram \n");
    int w = resolution[0];
    int h = resolution[1];
    int imod = 0;
    int c = CAMERAS;
    float s = SCALE;
    float r = RADIUS;
    float q  = QUOTIENT;
    int R, G, B;
    ALLEGRO_VERTEX_BUFFER* buffer = NULL;

    setDisplayVerticesBlack(lenV, vertices);

    for (int k = 0; k < W0; k++) { 
        for (int l = 0; l < W1; l++) {
            for (int i = 0; i < lenV; i++) {
                R = X[i][k*W1*3 + l*3 + 0];
                G = X[i][k*W1*3 + l*3 + 1];
                B = X[i][k*W1*3 + l*3 + 2];
                vertices[imod].color = al_map_rgb(R, G, B);
                imod++;                    
                vertices[imod].color = al_map_rgb(R, G, B);
                imod++;                    
                vertices[imod].color = al_map_rgb(R, G, B);
                imod++;
            }
        }
    }

    buffer = al_create_vertex_buffer(NULL, vertices, imod, 0);
    al_draw_vertex_buffer(buffer, NULL, 0, imod, ALLEGRO_PRIM_TRIANGLE_LIST);
    al_destroy_vertex_buffer(buffer);

    if (grid) {
        for (int k = 0; k < W0; k++) {
            for (int l = 0; l < W1; l++) {
                indexToHexagonPos(k, l, hexPos);
                al_draw_filled_circle(hexPos[0]*density+w/2, hexPos[1]*density+h/2, 1, al_map_rgb(255, 255, 255));
            }
        }
    }

}


int main(int argc, char **argv){
    printf("RADIUS %f\n", RADIUS);

    printf("HEIGHT %f\n", HEIGHT);

    ALLEGRO_DISPLAY *display = NULL;
    ALLEGRO_BITMAP *stamp = NULL;
    ALLEGRO_EVENT_QUEUE *event_queue = NULL;
    ALLEGRO_TIMER *timer = NULL;

    bool redraw = true;

    srand(time(NULL));

    float* triPos = (float*) malloc(sizeof(float) * 2);
    float* hexPos = (float*) malloc(sizeof(float) * 2);

    PROG_STATE* state = initializeState();

    state->P = generatePoints(state->lenP, state->maxP);
    state->Pc = generateColors(state->lenP, state->maxP);
    updateCursor(state);
    /*
    P[0][0] = 0;
    P[0][1] = 0;
    P[0][2] = 6;
    Pc[0][0] = 0;
    Pc[0][1] = 255;
    Pc[0][2] = 255;
    */
    

    printf("points generated\n"); 
    int** triDict = getTriDict(CAMERAS);
    int** triDictInv = getTriDictInv(CAMERAS);

    printf("triDicts generated \n");
    int lenV = getLenV(CAMERAS);
    printf("lenV: %d\n", lenV);
    float** V = generateViews(lenV, triDict);
    printf("V generated \n");

    int** X = allocHologram(lenV, W0, W1);
    printf("X generated \n");

    float* distances = allocDistances(W0, W1);
    ALLEGRO_VERTEX* vertices = allocVertices(W0, W1, lenV);
    initializeDisplayVertices(V, lenV, vertices, triDict, state->resolution, state->density, triPos, hexPos);

    float* projected = malloc(state->maxP * 2 * sizeof(float*));
    int* ind = malloc(state->maxP * 2 * sizeof(int*));




    printf("rest generated \n");


    if(!al_init_primitives_addon()) {
        fprintf(stderr, "failed to initialize allegro primitives!\n");
        return -1;
    }

    if(!al_init()) {
        fprintf(stderr, "failed to initialize allegro!\n");
        return -1;
    }
    if(!al_install_mouse()) {
        fprintf(stderr, "failed to initialize the mouse!\n");
        return -1;
    }
    if(!al_install_keyboard()) {
        fprintf(stderr, "failed to initialize the mouse!\n");
        return -1;
    }

    timer = al_create_timer(1.0 / state->FPS);
    if(!timer) {
        fprintf(stderr, "failed to create timer!\n");
        return -1;
    }

    display = al_create_display(state->resolution[0], state->resolution[1]);
    if(!display) {
        fprintf(stderr, "failed to create display!\n");
        return -1;
    }
    stamp = al_create_bitmap(state->resolution[0], state->resolution[1]);
    if(!stamp) {
        fprintf(stderr, "failed to create bouncer bitmap!\n");
        return -1;
    }
    al_set_target_bitmap(stamp);
    al_clear_to_color(al_map_rgb(0, 0, 0));
    al_set_target_bitmap(al_get_backbuffer(display));


    event_queue = al_create_event_queue();
    if(!event_queue) {
        fprintf(stderr, "failed to create event_queue!\n");
        return -1;
    }

    al_register_event_source(event_queue, al_get_display_event_source(display));
    al_register_event_source(event_queue, al_get_timer_event_source(timer));
    al_register_event_source(event_queue, al_get_mouse_event_source());
    al_register_event_source(event_queue, al_get_keyboard_event_source());

    ALLEGRO_EVENT ev;

    al_clear_to_color(al_map_rgb(0,0,0));
    al_flip_display();
    
    al_start_timer(timer);
    while(1)
    {
        al_wait_for_event(event_queue, &ev);

        if(ev.type == ALLEGRO_EVENT_TIMER) {

            redraw = true;
        }

        else if(ev.type == ALLEGRO_EVENT_MOUSE_AXES ||
                ev.type == ALLEGRO_EVENT_MOUSE_ENTER_DISPLAY) {

            handleMouseMove(ev.mouse, state);
        }

        else if(ev.type == ALLEGRO_EVENT_MOUSE_BUTTON_DOWN) {

            if (ev.mouse.button & 1) {
                handleLMBDown(ev.mouse, state);
            }
            else if (ev.mouse.button & 2) {
                handleRMBDown(ev.mouse, state);
            }

        }

        else if(ev.type == ALLEGRO_EVENT_KEY_DOWN) {
            printf("keydown\n");
            handleKeyboard(state, ev.keyboard.keycode);
            initializeDisplayVertices(V, lenV, vertices, triDict, state->resolution, state->density, triPos, hexPos);
         }

        else if(ev.type == ALLEGRO_EVENT_DISPLAY_CLOSE) {
            break;
        }

        

        if(redraw && al_is_event_queue_empty(event_queue)) {
            redraw = false;
            
            float timeA, timeB;
            timeA = clock();

            createHologram(state->P, state->Pc, state->lenP, V, lenV, X, distances, projected, ind);
            timeB = clock();
            //printf("createHolo: %f\n", (timeB - timeA) / CLOCKS_PER_SEC);
            al_clear_to_color(al_map_rgb(0,0,0));
            al_set_target_bitmap(stamp);
            
            displayHologram(X, V, lenV, vertices, triDict, state->resolution, state->density, state->grid, triPos, hexPos);
            timeA = clock();
        
            al_set_target_bitmap(al_get_backbuffer(display));
            
            al_draw_bitmap(stamp, 0, 0, 0);
            al_flip_display();
            //printf("displayHolo: %f\n", (timeA - timeB) / CLOCKS_PER_SEC);
        }
    }

    al_destroy_timer(timer);
    al_destroy_display(display);
    al_destroy_event_queue(event_queue);
    printf("shutdown 1\n");
    for(int i = 0; i < lenV; i++) {
        free(X[i]);
        free(V[i]);
    }
    printf("shutdown2\n");
    free(X);
    free(V);
    printf("shutdown 3\n");
    free(state->P);
    free(state->Pc);
    free(state);
    return (0);
}


float* generatePoints(int lenP, int maxP) {
    float low = -24;
    float high = 24;
    float height = 1;
    float* P = malloc(3 * maxP * sizeof(float));

    for (int n = 1; n < lenP; n++) {
        P[3*n] = low + ((float)rand() / RAND_MAX) * (high - low);
        P[3*n+1] = low + ((float)rand() / RAND_MAX) * (high - low);
        P[3*n+2] = height + ((float)rand() / RAND_MAX) * (high - 0);
    }
    return(P);
}

int* generateColors(int lenP, int maxP) {
    int* Pc = malloc(3 * maxP * sizeof(int));

    for (int n = 1; n < lenP; n++) {
        Pc[3*n] = ((float)rand() / RAND_MAX) * 255;
        Pc[3*n+1] = ((float)rand() / RAND_MAX) * 255;
        Pc[3*n+2] = ((float)rand() / RAND_MAX) * 255;
    }
    return(Pc);
}


void handleMouseMove(ALLEGRO_MOUSE_EVENT mouse, PROG_STATE* state) {
    //printf("P: %f %f %f\n", state->P[0], state->P[1], state->P[2]);
    //printf("P: %f %f %f\n", mouse.x - state->resolution[0]/2.0, mouse.y - state->resolution[1]/2.0, mouse.z);
    state->x = (mouse.x - state->resolution[0]/2.0) / state->density;
    state->y = (mouse.y - state->resolution[1]/2.0) / state->density;
    state->z += mouse.dz / state->density * 4;
    updateCursor(state);
    //printFloatArray(state->P, state->lenP * 3);
}

void addPoint(PROG_STATE* state, float x, float y, float z, int R, int G, int B) {
    if (state->lenP < state->maxP) {
        state->P[3*state->lenP + 0] = x;
        state->P[3*state->lenP + 1] = y;
        state->P[3*state->lenP + 2] = z;

        state->Pc[3*state->lenP + 0] = R;
        state->Pc[3*state->lenP + 1] = G;
        state->Pc[3*state->lenP + 2] = B;
        state->lenP += 1;
    }
    else {
        printf("maximum points reached\n");
    }
}

void handleLMBDown(ALLEGRO_MOUSE_EVENT mouse, PROG_STATE* state) {

    addPoint(state, state->x, state->y, state->z, state->color[0], state->color[1], state->color[2]);
     
}

void handleRMBDown(ALLEGRO_MOUSE_EVENT mouse, PROG_STATE* state) {
    float dist;
    float m;
    if (state->rmb_mode == 0) {
        state->prevx = state->x;
        state->prevy = state->y;
        state->prevz = state->z;

        state->rmb_mode = 1;
    }
    else {
        dist = sqrt(pow(state->x-state->prevx, 2) + pow(state->y-state->prevy, 2) + pow(state->z-state->prevz, 2));
        m = round(dist / state->line_density);
        for (int i = 0; i < (int) m; i++) {
            addPoint(state, state->prevx + i / m * (state->x-state->prevx),
                            state->prevy + i / m * (state->y-state->prevy),
                            state->prevz + i / m * (state->z-state->prevz),
                            state->color[0], state->color[1], state->color[2]);
        }

        state->rmb_mode = 0;
    }
}

void updateCursor(PROG_STATE* state) {
    state->P[0] = state->x;
    state->P[1] = state->y;
    state->P[2] = state->z;

    state->Pc[0] = state->color[0];
    state->Pc[1] = state->color[1];
    state->Pc[2] = state->color[2];
}

void handleColorChange(PROG_STATE* state, int R, int G, int B) {
    state->color[0] = R;
    state->color[1] = G;
    state->color[2] = B;
}

void handleSwitchSavingMode(PROG_STATE* state) {
    if (state->saving_mode) {
        state->saving_mode = 0;
    }
    else {
        state->saving_mode = 1;
    }
}

void handleSavePoints(PROG_STATE* state, char name) {
    char filename[3];
    filename[2] = '\0';
    FILE* f;
    int bfr[1];
    if (state->saving_mode) {

        bfr[0] = state->lenP;
        filename[0] = name;
        filename[1] = 'l'; 
        f = fopen(filename, "wb");
        fwrite(bfr, sizeof(int), 1, f);
        fclose(f);

        filename[0] = name;
        filename[1] = 'p'; 
        f = fopen(filename, "wb");
        fwrite(state->P, sizeof(float), state->lenP*3, f);
        fclose(f);

        filename[0] = name;
        filename[1] = 'c'; 
        f = fopen(filename, "wb");
        fwrite(state->Pc, sizeof(int), state->lenP*3, f);
        fclose(f);

    }
    else {

        filename[0] = name;
        filename[1] = 'l'; 
        f = fopen(filename, "rb");
        fread(bfr, sizeof(int), 1, f);
        fclose(f);
        printf("here\n");
        state->lenP = bfr[0];

        filename[0] = name;
        filename[1] = 'p'; 
        f = fopen(filename, "rb");
        fread(state->P, sizeof(float), state->lenP*3, f);
        fclose(f);

        filename[0] = name;
        filename[1] = 'c'; 
        f = fopen(filename, "rb");
        fread(state->Pc, sizeof(int), state->lenP*3, f);
        fclose(f);

    }

    state->saving_mode = 0;
}

void handleSwitchGrid(PROG_STATE* state) {
    if (state->grid) {
        state->grid = 0;
    }
    else {
        state->grid = 1;
    }
}

void handleChangeDensity(PROG_STATE* state, int direction) {
    if (direction) {
        state->density *= state->density_change;
    }
    else {
        state->density *= 1.0 / state->density_change;
    }
}

void handleChangeZoom(PROG_STATE* state, int direction) {
    if (direction) {
        for (int i = 0; i < 3 * state->lenP; i++) {
            state->P[i] *= state->zoom_change;
        }
    }
    else {
        for (int i = 0; i < 3 * state->lenP; i++) {
            state->P[i] *= 1.0 / state->zoom_change;
        }
    }
}


void handleKeyboard(PROG_STATE* state, int keycode) {
    switch(keycode) {
        case ALLEGRO_KEY_R:
            printf("R\n");
            handleColorChange(state, 255, 0, 0);
            break;

        case ALLEGRO_KEY_G:
            handleColorChange(state, 0, 255, 0);
            break;

        case ALLEGRO_KEY_B: 
            handleColorChange(state, 0, 0, 255);
            break;

        case ALLEGRO_KEY_M:
            handleColorChange(state, 255, 0, 255);
            break;

        case ALLEGRO_KEY_Y:
            handleColorChange(state, 255, 255, 0);
            break;

        case ALLEGRO_KEY_C: 
            handleColorChange(state, 0, 255, 255);
            break;

        case ALLEGRO_KEY_S:
            handleSwitchSavingMode(state);
            break;
        
        case ALLEGRO_KEY_T:
            handleSwitchGrid(state);;
            break;

        case ALLEGRO_KEY_U:
            handleChangeDensity(state, 0);
            break;

        case ALLEGRO_KEY_I:
            handleChangeDensity(state, 1);
            break;

        case ALLEGRO_KEY_K:
            handleChangeZoom(state, 0);
            break;

        case ALLEGRO_KEY_L:
            handleChangeZoom(state, 1);
            break;
        case ALLEGRO_KEY_X:
            state->lenP -= 5;
            break;

        case ALLEGRO_KEY_Z:
            state->lenP = 1;
            break;
        
        case ALLEGRO_KEY_1:
            handleSavePoints(state, '1');
            break;
        case ALLEGRO_KEY_2:
            handleSavePoints(state, '2');
            break;
        case ALLEGRO_KEY_3:
            handleSavePoints(state, '3');
            break;
        case ALLEGRO_KEY_4:
            handleSavePoints(state, '4');
            break;
        case ALLEGRO_KEY_5:
            handleSavePoints(state, '5');
            break;
        case ALLEGRO_KEY_6:
            handleSavePoints(state, '6');
            break;
        case ALLEGRO_KEY_7:
            handleSavePoints(state, '7');
            break;
        case ALLEGRO_KEY_8:
            handleSavePoints(state, '8');
            break;
        case ALLEGRO_KEY_9:
            handleSavePoints(state, '9');
            break;
    }
    updateCursor(state);
}

