#import "AppDelegate.h"
#include <iostream>
#include <sstream>
#include "MxDebug.h"
#include "MxEdge.h"



@implementation AppDelegate

@synthesize window = _windows;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    self->meshTest = new MeshTest();
    
    [self updateGuiFromModel];
    [self updateGuiStats];

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
    meshTest->step(0.0001);
    
    [self updateGuiStats];
}

-(IBAction)stop:(id)sender {
    [self.stepTimer invalidate];
    self.stepTimer = nil;
}


-(IBAction)reset:(id)sender {
    meshTest->reset();
    [self updateGuiFromModel];
    [self updateGuiStats];
}

-(IBAction)valueChanged:(id)sender {
    
   
    
    if (sender == self.cellCellSurfaceTensionSlider)
    {
        self.cellCellSurfaceTensionVal.floatValue = self.cellCellSurfaceTensionSlider.floatValue;
        meshTest->model->cellCellSurfaceTension = self.cellCellSurfaceTensionSlider.floatValue;
    }
    else if (sender == self.cellCellSurfaceTensionVal)
    {
        self.cellCellSurfaceTensionSlider.floatValue = self.cellCellSurfaceTensionVal.floatValue;
        meshTest->model->cellCellSurfaceTension = self.cellCellSurfaceTensionVal.floatValue;
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
    else if (sender == self.cellMediaSurfaceTensionVal)
    {
        self.cellMediaSurfaceTensionSlider.floatValue = self.cellMediaSurfaceTensionVal.floatValue;
        meshTest->model->cellMediaSurfaceTension = self.cellMediaSurfaceTensionVal.floatValue;
    }
    else if (sender == self.cellMediaSurfaceTensionSlider)
    {
        self.cellMediaSurfaceTensionVal.floatValue = self.cellMediaSurfaceTensionSlider.floatValue;
        meshTest->model->cellMediaSurfaceTension = self.cellMediaSurfaceTensionSlider.floatValue;
    }
    else if (sender == self.volumeVal)
    {
        self.volumeSlider.floatValue = self.volumeVal.floatValue;
        meshTest->model->setTargetVolume(self.volumeVal.floatValue);
    }
    else if (sender == self.volumeSlider)
    {
        self.volumeVal.floatValue = self.volumeSlider.floatValue;
        meshTest->model->setTargetVolume(self.volumeSlider.floatValue);
    }
    else if(sender == self.volumeLambda)
    {
        meshTest->model->targetVolumeLambda = self.volumeLambda.floatValue;
    }
    else if(sender == self.harmonicBondTxt)
    {
        meshTest->model->harmonicBondStrength = self.harmonicBondTxt.floatValue;
    }
    else if(sender == self.selectedEdgeSlider) {
        self.selectedEdgeVal.integerValue = self.selectedEdgeSlider.integerValue;
        meshTest->model->mesh->selectObject(MxEdge_Type, self.selectedEdgeSlider.integerValue);
    }
    else if(sender == self.selectedEdgeVal) {
        self.selectedEdgeSlider.integerValue = self.selectedEdgeVal.integerValue;
        meshTest->model->mesh->selectObject(MxEdge_Type, self.selectedEdgeVal.integerValue);
    }
    
    meshTest->draw();
    
    
    std::cout << "value changed, cellMediaSurfaceTension: " << meshTest->model->cellMediaSurfaceTension
    << ", surface tension: " << meshTest->model->cellCellSurfaceTension << std::endl;
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
    
    self.cellMediaSurfaceTensionMax.floatValue = meshTest->model->cellMediaSurfaceTensionMax;
    self.cellMediaSurfaceTensionMin.floatValue = meshTest->model->cellMediaSurfaceTensionMin;
    self.cellMediaSurfaceTensionVal.floatValue = meshTest->model->cellMediaSurfaceTension;
    
    self.cellMediaSurfaceTensionSlider.maxValue = meshTest->model->cellMediaSurfaceTensionMax;
    self.cellMediaSurfaceTensionSlider.minValue = meshTest->model->cellMediaSurfaceTensionMin;
    self.cellMediaSurfaceTensionSlider.floatValue = meshTest->model->cellMediaSurfaceTension;
    
    self.cellCellSurfaceTensionMax.floatValue = meshTest->model->cellCellSurfaceTensionMax;
    self.cellCellSurfaceTensionMin.floatValue = meshTest->model->cellCellSurfaceTensionMin;
    self.cellCellSurfaceTensionVal.floatValue = meshTest->model->cellCellSurfaceTension;
    
    self.cellCellSurfaceTensionSlider.maxValue = meshTest->model->cellCellSurfaceTensionMax;
    self.cellCellSurfaceTensionSlider.minValue = meshTest->model->cellCellSurfaceTensionMin;
    self.cellCellSurfaceTensionSlider.floatValue = meshTest->model->cellCellSurfaceTension;
    
    self.shortCutoff.floatValue = meshTest->model->mesh->getShortCutoff();
    self.longCutoff.floatValue = meshTest->model->mesh->getLongCutoff();
    
    self.constantVolumeBtn.state = meshTest->model->volumeForceType == GrowthModel::ConstantVolume ? NSOnState : NSOffState;
    
    self.volumeLambda.floatValue = meshTest->model->targetVolumeLambda;
    
    self.harmonicBondTxt.floatValue = meshTest->model->harmonicBondStrength;
    
    self.selectedEdgeMin.integerValue = 0;
    self.selectedEdgeMax.integerValue = meshTest->model->mesh->edges.size();
    self.selectedEdgeSlider.maxValue = meshTest->model->mesh->edges.size();
    self.selectedEdgeSlider.minValue = 0;
    
    EdgePtr e = (EdgePtr)meshTest->model->mesh->selectedObject();
    if (e) {
        self.selectedEdgeVal.integerValue = e->id;
        self.selectedEdgeSlider.integerValue = e->id;
    } else {
        self.selectedEdgeVal.integerValue = -1;
        self.selectedEdgeSlider.integerValue = -1;
    }
}

-(void)updateGuiStats {
    CCellPtr cell = meshTest->model->mesh->cells[1];
    
    self.centerOfGeometryTxt.stringValue =
        [NSString stringWithUTF8String:to_string(cell->centroid).c_str()];
    
    self.centerOfMassTxt.stringValue =
        [NSString stringWithUTF8String:to_string(cell->centerOfMass()).c_str()];
    
    Vector3 radius = cell->radiusMeanVarianceStdDev();
    
    self.radiusTxt.stringValue =
        [NSString stringWithUTF8String:to_string(radius).c_str()];
    
    Matrix3 inertia = cell->momentOfInertia();
    
    self.inertiaLxTxt.stringValue =
        [NSString stringWithUTF8String:to_string(inertia.row(0)).c_str()];
    
    self.inertiaLyTxt.stringValue =
        [NSString stringWithUTF8String:to_string(inertia.row(1)).c_str()];
    
    self.inertiaLzTxt.stringValue =
        [NSString stringWithUTF8String:to_string(inertia.row(2)).c_str()];
    
    self.actualVolumeTxt.floatValue = cell->volume;
    
    self.areaTxt.floatValue = cell->area;
}

-(IBAction)applyMeshOps:(id)sender {
    meshTest->model->mesh->applyMeshOperations();
    
}

-(IBAction)T1transitionSelectedEdge:(id)sender {
    HRESULT result = meshTest->model->applyT1Edge2TransitionToSelectedEdge();
    if(SUCCEEDED(result)) {
        std::cout << "successfully applied T1 transition" << std::endl;
    }
}





@end
