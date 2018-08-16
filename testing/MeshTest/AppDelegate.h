// Hideously ugly include order, has to be this way because of Magnum gl include wierdness. 

typedef unsigned char Byte;
#import <AppKit/NSApplication.h> // NSApplicationDelegate
#include "MeshTest.h"
#import <Cocoa/Cocoa.h>


@interface AppDelegate : NSObject <NSApplicationDelegate> {
    MeshTest *meshTest;
    float radius;
}

@property (assign, atomic) NSTimer *stepTimer; 
@property (assign, nonatomic) IBOutlet NSWindow *window;

@property (assign, nonatomic) IBOutlet NSButton *constantVolumeBtn;
@property (assign, nonatomic) IBOutlet NSButton *constantPressureBtn;


@property (assign, nonatomic) IBOutlet NSSlider *cellMediaSurfaceTensionSlider;
@property (assign, nonatomic) IBOutlet NSTextField *cellMediaSurfaceTensionMin;
@property (assign, nonatomic) IBOutlet NSTextField *cellMediaSurfaceTensionMax;
@property (assign, nonatomic) IBOutlet NSTextField *cellMediaSurfaceTensionVal;


@property (assign, nonatomic) IBOutlet NSSlider *cellCellSurfaceTensionSlider;
@property (assign, nonatomic) IBOutlet NSTextField *cellCellSurfaceTensionMin;
@property (assign, nonatomic) IBOutlet NSTextField *cellCellSurfaceTensionMax;
@property (assign, nonatomic) IBOutlet NSTextField *cellCellSurfaceTensionVal;

@property (assign, nonatomic) IBOutlet NSSlider *volumeSlider;
@property (assign, nonatomic) IBOutlet NSTextField *volumeMin;
@property (assign, nonatomic) IBOutlet NSTextField *volumeMax;
@property (assign, nonatomic) IBOutlet NSTextField *volumeVal;

@property (assign, nonatomic) IBOutlet NSTextField *shortCutoff;
@property (assign, nonatomic) IBOutlet NSTextField *longCutoff;

@property (assign, nonatomic) IBOutlet NSTextField *volumeLambda;
@property (assign) IBOutlet NSTextField *centerOfGeometryTxt;
@property (assign) IBOutlet NSTextField *centerOfMassTxt;
@property (assign) IBOutlet NSTextField *radiusTxt;
@property (assign) IBOutlet NSTextField *inertiaLxTxt;
@property (assign) IBOutlet NSTextField *inertiaLyTxt;
@property (assign) IBOutlet NSTextField *inertiaLzTxt;
@property (assign) IBOutlet NSTextField *actualVolumeTxt;
@property (assign) IBOutlet NSTextField *areaTxt;
@property (assign) IBOutlet NSTextField *harmonicBondTxt;

-(IBAction)run:(id)sender;

-(IBAction)step:(id)sender;

-(IBAction)stop:(id)sender;

-(IBAction)reset:(id)sender;

-(IBAction)valueChanged:(id)sender;

-(IBAction)volumeForceClick:(id)sender;

-(IBAction)applyMeshOps:(id)sender;

-(void)updateGuiFromModel;

-(void)updateGuiStats;






@end
