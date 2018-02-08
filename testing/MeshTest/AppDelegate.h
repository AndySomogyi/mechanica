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


@property (assign, nonatomic) IBOutlet NSSlider *pressureSlider;
@property (assign, nonatomic) IBOutlet NSTextField *pressureMin;
@property (assign, nonatomic) IBOutlet NSTextField *pressureMax;
@property (assign, nonatomic) IBOutlet NSTextField *pressureVal;


@property (assign, nonatomic) IBOutlet NSSlider *surfaceTensionSlider;
@property (assign, nonatomic) IBOutlet NSTextField *surfaceTensionMin;
@property (assign, nonatomic) IBOutlet NSTextField *surfaceTensionMax;
@property (assign, nonatomic) IBOutlet NSTextField *surfaceTensionVal;

@property (assign, nonatomic) IBOutlet NSSlider *volumeSlider;
@property (assign, nonatomic) IBOutlet NSTextField *volumeMin;
@property (assign, nonatomic) IBOutlet NSTextField *volumeMax;
@property (assign, nonatomic) IBOutlet NSTextField *volumeVal;

@property (assign, nonatomic) IBOutlet NSTextField *shortCutoff;
@property (assign, nonatomic) IBOutlet NSTextField *longCutoff;

@property (assign, nonatomic) IBOutlet NSTextField *volumeLambda;

-(IBAction)run:(id)sender;

-(IBAction)step:(id)sender;

-(IBAction)stop:(id)sender;

-(IBAction)reset:(id)sender;

-(IBAction)valueChanged:(id)sender;

-(IBAction)volumeForceClick:(id)sender;

-(void)updateGuiFromModel;




@end
