/* Small C file creating an array to demo C -> Python data passing 
 * 
 * Author: Gael Varoquaux
 * License: BSD
 */

#include <stdlib.h>

double *compute(int size)
{
    double* array;
    array = malloc(sizeof(double)*size);
    for (int i=0; i<size; i++)
    {
        // array[i] = (double)i;
        array[i] = (double)i;
    }
    return array;
}

