// main.h
#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

void say_hello_from_cpp();  // callable from C
void do_get_request();
void do_post_request();
void opencv_image_encode();

#ifdef __cplusplus
}
#endif

#endif // MAIN_H
