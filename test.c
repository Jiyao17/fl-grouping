/******************************************************************************

                            Online C Compiler.
                Code, Compile, Run and Debug C program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <stdio.h> 

int get_material(double bp, double range);

int main()
{  
    double bp; 
    printf("Enter the boiling point: "); 
    scanf("%lf", &bp); 
    printf("%lf", bp);

    
    char* names[] = {"water", "mercury", "copper", "silver", "gold"}; 
    int index = get_material(bp, 0.1);  
    printf("%d", index);
    if(index != -1) 
        printf("It is %s.", names[index]); 
    else  
        printf("Unknown material");

    return 0;
} 

int get_material(double bp, double range) 
{ 
    printf("inside function");
    double bps[] = {100, 357, 1187, 2193, 2660}; 
    if(bps[0] * (1-range) <= bp && bp <= bps[0]*(1+range)) 
    // 90 <= bp && bp <= 110 
    { 
        return 0; 
    }  
    if(bps[1] * (1-range) <= bp && bp <= bps[1]*(1+range)) 
    // 357 - 35.7  <= bp && bp <= 357 + 35.7
    { 
        return 1; 
    }  
    if(bps[2] * (1-range) <= bp && bp <= bps[2]*(1+range)) 
    // 1187 - 11.87 <= bp && bp <= 1187 + 11.87 
    { 
        return 2; 
    }    
    if(bps[3] * (1-range) <= bp && bp <= bps[3]*(1+range))   
    {
        return 3;   
    } 
    if(bps[4] * (1-range) <= bp && bp <= bps[4]*(1+range))  
    { 
        return 4; 
    } 
    
    return -1;
}
    
    
    

