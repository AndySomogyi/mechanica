typedef float (*rk_dxdt)(float x);


float rk1_ti_step ( float x, float t, float h, float q,
  rk_dxdt fi, rk_dxdt gi, int *seed );

float rk2_ti_step ( float x, float t, float h, float q,
  float fi ( float x ), float gi ( float x ), int *seed );
float rk3_ti_step ( float x, float t, float h, float q,
  float fi ( float x ), float gi ( float x ), int *seed );
float rk4_ti_step ( float x, float t, float h, float q,
  float fi ( float x ), float gi ( float x ), int *seed );

float rk1_tv_step ( float x, float t, float h, float q,
  float fv ( float t, float x ), float gv ( float t, float x ),
  int *seed );
float rk2_tv_step ( float x, float t, float h, float q,
  float fv ( float t, float x ), float gv ( float t, float x ),
  int *seed );
float rk4_tv_step ( float x, float t, float h, float q,
  float fv ( float t, float x ), float gv ( float t, float x ),
  int *seed );

float r8_normal_01 ( int *seed );
float r8_uniform_01 ( int *seed );
void timestamp ( );
