// main.c
#include <stdio.h>
#include "main.h"

int main() {
    printf("Hello from C!\n");
    say_hello_from_cpp();

    printf("get from C!\n");
    do_get_request();

    printf("post from C!\n");
    do_post_request();
    return 0;
}
