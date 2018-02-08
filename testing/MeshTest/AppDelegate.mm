#import "AppDelegate.h"
#include <iostream>

static float pressure_from_volume(float volume) {
    // v = 4/3 pi r ^3
    // (3/(4 pi) v)^(1/3)
    
    return std::cbrt((3. / (4.*M_PI)) * volume);
}

static float volume(float pressure) {
    return 4. / 3. * M_PI * pressure * pressure * pressure;
    
}

static float surfaceTension(float pressure) {
    return 1 * M_PI * pressure * pressure;
}




@implementation AppDelegate

@synthesize window = _windows;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    self->meshTest = new MeshTest();
    
    [self updateGuiFromModel];
    

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
    [self updateGuiFromModel];

}

-(IBAction)valueChanged:(id)sender {
    
   
    
    if (sender == self.surfaceTensionSlider)
    {
        self.surfaceTensionVal.floatValue = self.surfaceTensionSlider.floatValue;
        meshTest->model->surfaceTension = self.surfaceTensionSlider.floatValue;
    }
    else if (sender == self.surfaceTensionVal)
    {
        self.surfaceTensionSlider.floatValue = self.surfaceTensionVal.floatValue;
        meshTest->model->surfaceTension = self.surfaceTensionVal.floatValue;
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
    else if (sender == self.pressureVal)
    {
        self.pressureSlider.floatValue = self.pressureVal.floatValue;
        meshTest->model->pressure = self.pressureVal.floatValue;
    }
    else if (sender == self.pressureSlider)
    {
        self.pressureVal.floatValue = self.pressureSlider.floatValue;
        meshTest->model->pressure = self.pressureSlider.floatValue;
    }
    else if (sender == self.volumeVal)
    {
        self.volumeSlider.floatValue = self.volumeVal.floatValue;
        meshTest->model->targetVolume = self.volumeVal.floatValue;
    }
    else if (sender == self.volumeSlider)
    {
        self.volumeVal.floatValue = self.volumeSlider.floatValue;
        meshTest->model->targetVolume = self.volumeSlider.floatValue;
    }
    else if(sender == self.volumeLambda)
    {
        meshTest->model->targetVolumeLambda = self.volumeLambda.floatValue;
    }
    
    std::cout << "value changed, pressure: " << meshTest->model->pressure
    << ", surface tension: " << meshTest->model->surfaceTension << std::endl;
}

-(IBAction)volumeForceClick:(id)sender {
    if(sender == self.constantVolumeBtn) {
        self->meshTest->model->volumeForceType = GrowthModel::ConstantVolume;
    }
    
    else if(sender == self.constantPressureBtn) {
        self->meshTest->model->volumeForceType = GrowthModel::ConstantPressure;
    }
}

-(void)updateGuiFromModel {
    self.volumeMax.floatValue = meshTest->model->maxTargetVolume;
    self.volumeMin.floatValue = meshTest->model->minTargetVolume;
    self.volumeVal.floatValue = meshTest->model->targetVolume;
    
    self.volumeSlider.maxValue = meshTest->model->maxTargetVolume;
    self.volumeSlider.minValue = meshTest->model->minTargetVolume;
    self.volumeSlider.floatValue = meshTest->model->targetVolume;
    
    self.pressureMax.floatValue = meshTest->model->pressureMax;
    self.pressureMin.floatValue = meshTest->model->pressureMin;
    self.pressureVal.floatValue = meshTest->model->pressure;
    
    self.pressureSlider.maxValue = meshTest->model->pressureMax;
    self.pressureSlider.minValue = meshTest->model->pressureMin;
    self.pressureSlider.floatValue = meshTest->model->pressure;
    
    self.surfaceTensionMax.floatValue = meshTest->model->surfaceTensionMax;
    self.surfaceTensionMin.floatValue = meshTest->model->surfaceTensionMin;
    self.surfaceTensionVal.floatValue = meshTest->model->surfaceTension;
    
    self.surfaceTensionSlider.maxValue = meshTest->model->surfaceTensionMax;
    self.surfaceTensionSlider.minValue = meshTest->model->surfaceTensionMin;
    self.surfaceTensionSlider.floatValue = meshTest->model->surfaceTension;
    
    self.shortCutoff.floatValue = meshTest->model->mesh->getShortCutoff();
    self.longCutoff.floatValue = meshTest->model->mesh->getLongCutoff();
    
    self.constantVolumeBtn.state = meshTest->model->volumeForceType == GrowthModel::ConstantVolume ? NSOnState : NSOffState;
    
    self.volumeLambda.floatValue = meshTest->model->targetVolumeLambda;
}





@end
