#import "AppDelegate.h"
#include <iostream>


@implementation AppDelegate

@synthesize window = _windows;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    self->meshTest = new MeshTest();
    
    self.areaMax.doubleValue = meshTest->model->maxTargetArea;
    self.areaMin.doubleValue = meshTest->model->minTargetArea;
    self.areaVal.doubleValue = meshTest->model->targetArea;
    self.areaSlider.floatValue = meshTest->model->targetArea;
    self.areaSlider.maxValue = meshTest->model->maxTargetArea;
    self.areaSlider.minValue = meshTest->model->minTargetArea;
}


-(IBAction)run:(id)sender {
    
    if(self.stepTimer) {
        return;
    }
    
    [self.stepTimer invalidate];
    
    NSTimer *timer = [NSTimer scheduledTimerWithTimeInterval:0.1
                              target:self selector:@selector(step:)
                              userInfo:nil repeats:YES];
    self.stepTimer = timer;
}

-(IBAction)step:(id)sender {
    meshTest->step(0.1);
}

-(IBAction)stop:(id)sender {
    [self.stepTimer invalidate];
    self.stepTimer = nil;
}

- (IBAction)areaValueChanged:(id)sender {
    NSLog(@"Value Changed");
    
    if (sender == self.areaSlider)
    {
        self.areaVal.floatValue = self.areaSlider.floatValue;
        meshTest->model->targetArea = self.areaSlider.floatValue;
    }
    else
    {
        self.areaSlider.floatValue = self.areaVal.floatValue;
        meshTest->model->targetArea = self.areaVal.floatValue;
    }
}

-(IBAction)reset:(id)sender {
    meshTest->reset();
}

@end
