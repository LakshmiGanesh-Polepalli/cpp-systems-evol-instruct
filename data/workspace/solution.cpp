#include <iostream>

int main() {
    int fib[10] = {0, 1};
    
    for (int i = 2; i < 10; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    
    for (int i = 0; i < 10; ++i) {
        std::cout << fib[i] << " ";
    }
    
    return 0;
}