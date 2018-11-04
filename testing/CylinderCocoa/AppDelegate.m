#import "AppDelegate.h"


@implementation AppDelegate

-(id)init
{
    if (self = [super init])
    {
        obj = NULL;
    }
    
    
    return self;
}


- (IBAction)testClick:(id)sender {
    CylinderTest_Create(600, 800, &obj);
    
    CylinderTest_LoadMesh(obj, "/Users/andy/src/mechanica/testing/models/hex_cylinder.1.obj");
}



@end
