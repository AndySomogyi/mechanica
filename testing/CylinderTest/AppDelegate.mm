#import "AppDelegate.h"
#include <iostream>
#include <sstream>
#include "MxDebug.h"
#include "MxEdge.h"
#include "MeshIO.h"

void testIndexOf() {
    std::vector<int> numbers = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
    
    std::cout << "size of numbers: " << numbers.size() << std::endl;
    for (int i = -5; i < 15; ++i) {
        std::cout << "index of " << i << ": " << indexOf(numbers, i) << std::endl;
    }
}



@implementation AppDelegate

@synthesize window = _windows;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    self->meshTest = new CylinderTest();
    
    [self updateGuiFromModel];
    [self updateGuiStats];
    
    testIndexOf();
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
    meshTest->loadModel();
    [self updateGuiFromModel];
    [self updateGuiStats];
}

-(IBAction)valueChanged:(id)sender {
    
    if (sender == self.areaSlider)
    {
        self.areaVal.floatValue = self.areaSlider.floatValue;
        meshTest->model->setTargetArea(self.areaSlider.floatValue);
    }
    else if (sender == self.areaVal)
    {
        self.areaSlider.floatValue = self.areaVal.floatValue;
        meshTest->model->setTargetArea(self.areaVal.floatValue);
    }
    else if (sender == self.cellMediaSurfaceTensionVal)
    {
        self.cellMediaSurfaceTensionSlider.floatValue = self.cellMediaSurfaceTensionVal.floatValue;
        meshTest->model->setStdSurfaceTension(self.cellMediaSurfaceTensionVal.floatValue);
    }
    else if (sender == self.cellMediaSurfaceTensionSlider)
    {
        self.cellMediaSurfaceTensionVal.floatValue = self.cellMediaSurfaceTensionSlider.floatValue;
        meshTest->model->setStdSurfaceTension(self.cellMediaSurfaceTensionSlider.floatValue);
    }
    
    else if (sender == self.growingSurfaceTensionVal)
    {
        self.growingSurfaceTensionSlider.floatValue = self.growingSurfaceTensionVal.floatValue;
        meshTest->model->setGrowStdSurfaceTension(self.growingSurfaceTensionVal.floatValue);
    }
    else if (sender == self.growingSurfaceTensionSlider)
    {
        self.growingSurfaceTensionVal.floatValue = self.growingSurfaceTensionSlider.floatValue;
        meshTest->model->setGrowStdSurfaceTension(self.growingSurfaceTensionSlider.floatValue);
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
        meshTest->model->setTargetVolumeLambda(self.volumeLambda.floatValue);
    }
    else if(sender == self.selectedEdgeSlider) {
        self.selectedEdgeVal.integerValue = self.selectedEdgeSlider.integerValue;
        [self.selectableEdgeToTriEdges selectItemWithTitle:self.selectedEdgeSlider.stringValue];
        meshTest->model->mesh->selectObject(MxEdge_Type, self.selectedEdgeSlider.integerValue);
    }
    else if(sender == self.selectedEdgeVal) {
        self.selectedEdgeSlider.integerValue = self.selectedEdgeVal.integerValue;
        [self.selectableEdgeToTriEdges selectItemWithTitle:self.selectedEdgeVal.stringValue];
        meshTest->model->mesh->selectObject(MxEdge_Type, self.selectedEdgeVal.integerValue);
    }
    else if(sender == self.selectedPolygonSlider) {
        self.selectedPolygonVal.integerValue = self.selectedPolygonSlider.integerValue;
        [self selectChanged];
    }
    else if(sender == self.selectedPolygonVal) {
        self.selectedPolygonSlider.integerValue = self.selectedPolygonVal.integerValue;
        [self selectChanged];
    }
    else if(sender == self.selectableEdgeToTriEdges) {
        NSString *s = self.selectableEdgeToTriEdges.selectedItem.title;
        meshTest->model->mesh->selectObject(MxEdge_Type, s.integerValue);
        self.selectedEdgeSlider.integerValue = s.integerValue;
        self.selectedEdgeVal.integerValue = s.integerValue;
    }
    
    meshTest->draw();
    
    std::cout << "value changed, cellMediaSurfaceTension: " << meshTest->model->stdSurfaceTension() << std::endl;
}

-(IBAction)volumeForceClick:(id)sender {
}

-(void)updateGuiFromModel {
    self.volumeMax.floatValue = meshTest->model->maxTargetVolume();
    self.volumeMin.floatValue = meshTest->model->minTargetVolume();
    self.volumeVal.floatValue = meshTest->model->targetVolume();
    
    self.volumeSlider.maxValue = meshTest->model->maxTargetVolume();
    self.volumeSlider.minValue = meshTest->model->minTargetVolume();
    self.volumeSlider.floatValue = meshTest->model->targetVolume();
    
    self.cellMediaSurfaceTensionMax.floatValue = meshTest->model->stdSurfaceTensionMax();
    self.cellMediaSurfaceTensionMin.floatValue = meshTest->model->stdSurfaceTensionMin();
    self.cellMediaSurfaceTensionVal.floatValue = meshTest->model->stdSurfaceTension();
    
    self.cellMediaSurfaceTensionSlider.maxValue = meshTest->model->stdSurfaceTensionMax();
    self.cellMediaSurfaceTensionSlider.minValue = meshTest->model->stdSurfaceTensionMin();
    self.cellMediaSurfaceTensionSlider.floatValue = meshTest->model->stdSurfaceTension();
    
    self.growingSurfaceTensionMax.floatValue = meshTest->model->growSurfaceTensionMax();
    self.growingSurfaceTensionMin.floatValue = meshTest->model->growSurfaceTensionMin();
    self.growingSurfaceTensionVal.floatValue = meshTest->model->growSurfaceTension();
    
    self.growingSurfaceTensionSlider.maxValue = meshTest->model->growSurfaceTensionMax();
    self.growingSurfaceTensionSlider.minValue = meshTest->model->growSurfaceTensionMin();
    self.growingSurfaceTensionSlider.floatValue = meshTest->model->growSurfaceTension();
    
    self.areaMax.floatValue = meshTest->model->maxTargetArea();
    self.areaMin.floatValue = meshTest->model->minTargetArea();
    self.areaVal.floatValue = meshTest->model->targetArea();
    
    self.areaSlider.maxValue = meshTest->model->maxTargetArea();
    self.areaSlider.minValue = meshTest->model->minTargetArea();
    self.areaSlider.floatValue = meshTest->model->targetArea();
    
    self.volumeLambda.floatValue = meshTest->model->targetVolumeLambda();
    
    self.selectedEdgeMin.integerValue = 0;
    self.selectedEdgeMax.integerValue = meshTest->model->mesh->edges.size();
    self.selectedEdgeSlider.maxValue = meshTest->model->mesh->edges.size();
    self.selectedEdgeSlider.minValue = 0;
    
    self.selectedPolygonMin.integerValue = 0;
    self.selectedPolygonMax.integerValue = meshTest->model->mesh->polygons.size();
    self.selectedPolygonSlider.maxValue = meshTest->model->mesh->polygons.size();
    self.selectedPolygonSlider.minValue = 0;
    
    [self.selectableEdgeToTriEdges removeAllItems];
    for(int i = 0; i < meshTest->model->mesh->edges.size(); ++i) {
        EdgePtr e = meshTest->model->mesh->edges[i];
        if(Mx_IsEdgeToTriangleConfiguration(e)) {
            NSString *str = [NSString stringWithFormat:@"%i", e->id];
            [self.selectableEdgeToTriEdges addItemWithTitle:str];
        }
    }
    
    MxObject *obj = (EdgePtr)meshTest->model->mesh->selectedObject();
    
    if(obj) {
        EdgePtr e = dyn_cast<MxEdge>(obj);
        if (e) {
            self.selectedEdgeVal.integerValue = e->id;
            self.selectedEdgeSlider.integerValue = e->id;
            [self.selectableEdgeToTriEdges selectItemWithTitle:[NSString stringWithFormat:@"%i", e->id]];
            
            //std::cout << "seelct item value: " << [self.selectableEdgeToTriEdges.selectedItem.title UTF8String] << std::endl;
            
            selectType = MxEdge_Type;
        }
        
        PolygonPtr p = dyn_cast<MxPolygon>(obj);
        if (p) {
            self.selectedPolygonVal.integerValue = p->id;
            self.selectedPolygonSlider.integerValue = p->id;
            selectType = MxPolygon_Type;
        }
    }
    
    [self selectChanged];
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
    
    std::vector<int> v = {{0, 1, 2, 3, 4, 5}};
    
    v.insert(v.begin() + 6, 23);
    
    for(int i : v) {
        std::cout << i << std::endl;
    }
    
    meshTest->draw();
}

-(id)init
{
    if (self = [super init])
    {
        // Initialization code here
        selectType = MxEdge_Type;
        meshTest = nullptr;
    }
    
    
    return self;
}

-(IBAction)selectClicked:(NSPopUpButton*)sender {
    NSString *title = sender.selectedItem.title;
    
    std::string name{title.UTF8String};
    
    if(name == "Edges") {
        selectType = MxEdge_Type;

    }
    else if(name == "Polygons") {
        selectType = MxPolygon_Type;
    }
    
    [self selectChanged];
}

-(void)selectChanged {
    if(selectType == MxEdge_Type) {
        self.selectedPolygonMax.enabled = false;
        self.selectedPolygonMin.enabled = false;
        self.selectedPolygonVal.enabled = false;
        self.selectedPolygonSlider.enabled = false;
        
        self.selectedEdgeMax.enabled = true;
        self.selectedEdgeMin.enabled = true;
        self.selectedEdgeVal.enabled = true;
        self.selectedEdgeSlider.enabled = true;
        
        if(meshTest) {
            meshTest->model->mesh->selectObject(MxEdge_Type, self.selectedEdgeVal.integerValue);
            meshTest->draw();
        }
    }
    else if(selectType == MxPolygon_Type) {
        self.selectedPolygonMax.enabled = true;
        self.selectedPolygonMin.enabled = true;
        self.selectedPolygonVal.enabled = true;
        self.selectedPolygonSlider.enabled = true;
        
        self.selectedEdgeMax.enabled = false;
        self.selectedEdgeMin.enabled = false;
        self.selectedEdgeVal.enabled = false;
        self.selectedEdgeSlider.enabled = false;
        
        if(meshTest) {
            meshTest->model->mesh->selectObject(MxPolygon_Type, self.selectedPolygonVal.integerValue);
            meshTest->draw();
        }
    }
}

-(IBAction)awakeFromNib {
    std::cout << "awake" << std::endl;
    
    NSNumberFormatter *f = self.cellMediaSurfaceTensionVal.formatter;
    
    std::cout << "frac digits: " << f.maximumFractionDigits << std::endl;
    
    f.maximumFractionDigits = 10;
    
    std::cout << "frac digits: " << f.maximumFractionDigits << std::endl;
    
    [self selectChanged];
}

-(IBAction)T2transitionSelectedPolygon:(id)sender {
    HRESULT result = meshTest->model->applyT2PolygonTransitionToSelectedPolygon();
    
    if(SUCCEEDED(result)) {
        std::cout << "successfully applied T2 transition" << std::endl;
    }
    
    [self updateGuiFromModel];
    
    meshTest->draw();
}

-(IBAction)T3transitionSelectedPolygon:(id)sender {
    
    HRESULT result = meshTest->model->applyT3PolygonTransitionToSelectedPolygon();
    
    if(SUCCEEDED(result)) {
        std::cout << "successfully applied T2 transition" << std::endl;
    }
    
    [self updateGuiFromModel];
    
    meshTest->draw();
}

- (IBAction)saveDocument:(id)sender {
    MxMesh_WriteFile(meshTest->model->mesh, "foo");
}

-(IBAction)changePolygonTypes:(id)sender {
    HRESULT result = meshTest->model->changePolygonTypes();
    
    [self updateGuiFromModel];
    
    if(SUCCEEDED(result)) {
        std::cout << "successfully changed polygon type" << std::endl;
    }
    else {
        std::cout << "failed to changed polygon type" << std::endl;
    }
    
    meshTest->draw();
}

-(IBAction)activateAreaConstraint:(id)sender {
    HRESULT result = meshTest->model->activateAreaConstraint();
    
    [self updateGuiFromModel];
    
    if(SUCCEEDED(result)) {
        std::cout << "successfully activated area constraint type" << std::endl;
    }
    else {
        std::cout << "failed to activate area constraint" << std::endl;
    }
    
    meshTest->draw();
}

-(IBAction)edgeToPolygonFlip:(id)sender {
    HRESULT result = meshTest->model->edgeToPolygonFlipSelecgtedEdge();
    meshTest->draw();
    
    [self updateGuiFromModel];
    
    if(SUCCEEDED(result)) {
        std::cout << "successfully flipped edge" << std::endl;
    }
    else {
        std::cout << "failed to flip edge" << std::endl;
    }
    
}






@end
