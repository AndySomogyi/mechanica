#import "AppDelegate.h"
#include <iostream>

static float radius_from_volume(float volume) {
    // v = 4/3 pi r ^3
    // (3/(4 pi) v)^(1/3)
    
    return std::cbrt((3. / (4.*M_PI)) * volume);
}

static float volume(float radius) {
    return 4. / 3. * M_PI * radius * radius * radius;
    
}

static float area(float radius) {
    return 1 * M_PI * radius * radius;
}




@implementation AppDelegate

@synthesize window = _windows;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    self->meshTest = new MeshTest();
    
    self.radiusMax.floatValue = radius_from_volume(meshTest->model->maxTargetVolume);
    self.radiusMin.floatValue = radius_from_volume(meshTest->model->minTargetVolume);
    self.radiusVal.floatValue = radius_from_volume(meshTest->model->targetVolume);
    self.radiusSlider.floatValue = radius_from_volume(meshTest->model->targetVolume);
    self.radiusSlider.maxValue = radius_from_volume(meshTest->model->maxTargetVolume);
    self.radiusSlider.minValue = radius_from_volume(meshTest->model->minTargetVolume);
    
    meshTest->model->targetVolume = volume(self.radiusVal.floatValue);
    meshTest->model->targetArea = area(self.radiusVal.floatValue);
    
    self.areaMax.floatValue = meshTest->model->maxTargetArea;
    self.areaMin.floatValue = meshTest->model->minTargetArea;
    self.areaVal.floatValue = meshTest->model->targetArea;
    self.areaSlider.floatValue = meshTest->model->targetArea;
    self.areaSlider.maxValue = meshTest->model->maxTargetArea;
    self.areaSlider.minValue = meshTest->model->minTargetArea;
    
    self.volumeMax.floatValue = meshTest->model->maxTargetVolume;
    self.volumeMin.floatValue = meshTest->model->minTargetVolume;
    self.volumeVal.floatValue = meshTest->model->targetVolume;
    self.volumeSlider.floatValue = meshTest->model->targetVolume;
    self.volumeSlider.maxValue = meshTest->model->maxTargetVolume;
    self.volumeSlider.minValue = meshTest->model->minTargetVolume;
    

    
    self.shortCutoff.floatValue = meshTest->model->mesh->getShortCutoff();
    self.longCutoff.floatValue = meshTest->model->mesh->getLongCutoff();
}


-(IBAction)run:(id)sender {
    
    if(self.stepTimer) {
        return;
    }
    
    [self.stepTimer invalidate];
    
    NSTimer *timer = [NSTimer scheduledTimerWithTimeInterval:0.00001
                              target:self selector:@selector(step:)
                              userInfo:nil repeats:YES];
    self.stepTimer = timer;
}

-(IBAction)step:(id)sender {
    meshTest->step(0.00001);
}

-(IBAction)stop:(id)sender {
    [self.stepTimer invalidate];
    self.stepTimer = nil;
}


-(IBAction)reset:(id)sender {
    meshTest->reset();
}

-(IBAction)valueChanged:(id)sender {
    
    NSLog(@"Value Changed");
    
    if (sender == self.areaSlider)
    {
        self.areaVal.floatValue = self.areaSlider.floatValue;
        meshTest->model->targetArea = self.areaSlider.floatValue;
    }
    else if (sender == self.areaVal)
    {
        self.areaSlider.floatValue = self.areaVal.floatValue;
        meshTest->model->targetArea = self.areaVal.floatValue;
    }
    else if (sender == self.shortCutoff)
    {
        meshTest->model->mesh->setShortCutoff(self.shortCutoff.floatValue);
        meshTest->model->mesh->applyMeshOperations();
        meshTest->draw();
    }
    else if (sender == self.longCutoff)
    {
        meshTest->model->mesh->setLongCutoff(self.longCutoff.floatValue);
        meshTest->model->mesh->applyMeshOperations();
        meshTest->draw();
    }
    else if (sender == self.volumeVal)
    {
        self.volumeSlider.floatValue = self.volumeVal.floatValue;
        meshTest->model->targetVolume = self.volumeVal.floatValue;
        self.radiusVal.floatValue = radius_from_volume(meshTest->model->targetVolume);
        self.radiusSlider.floatValue = radius_from_volume(meshTest->model->targetVolume);

    }
    else if (sender == self.volumeSlider)
    {
        self.volumeVal.floatValue = self.volumeSlider.floatValue;
        meshTest->model->targetVolume = self.volumeSlider.floatValue;
        self.radiusVal.floatValue = radius_from_volume(meshTest->model->targetVolume);
        self.radiusSlider.floatValue = radius_from_volume(meshTest->model->targetVolume);
    }
    else if (sender == self.radiusVal)
    {
        self.radiusSlider.floatValue = self.radiusVal.floatValue;
        self.volumeSlider.floatValue = volume(self.radiusVal.floatValue);
        self.volumeVal.floatValue = volume(self.radiusVal.floatValue);
        meshTest->model->targetVolume = volume(self.radiusVal.floatValue);
        self.areaSlider.floatValue = area(self.radiusVal.floatValue);
        self.areaVal.floatValue = area(self.radiusVal.floatValue);
        meshTest->model->targetArea = area(self.radiusVal.floatValue);
    }
    else if (sender == self.radiusSlider)
    {
        self.radiusVal.floatValue = self.radiusSlider.floatValue;
        self.volumeSlider.floatValue = volume(self.radiusSlider.floatValue);
        self.volumeVal.floatValue = volume(self.radiusSlider.floatValue);
        meshTest->model->targetVolume = volume(self.radiusSlider.floatValue);
        self.areaSlider.floatValue = area(self.radiusSlider.floatValue);
        self.areaVal.floatValue = area(self.radiusSlider.floatValue);
        meshTest->model->targetArea = area(self.radiusSlider.floatValue);
    }
}

@end
