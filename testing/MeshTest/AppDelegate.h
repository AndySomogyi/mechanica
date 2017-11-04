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


@property (assign, nonatomic) IBOutlet NSSlider *areaSlider;
@property (assign, nonatomic) IBOutlet NSTextField *areaMin;
@property (assign, nonatomic) IBOutlet NSTextField *areaMax;
@property (assign, nonatomic) IBOutlet NSTextField *areaVal;


@property (assign, nonatomic) IBOutlet NSSlider *volumeSlider;
@property (assign, nonatomic) IBOutlet NSTextField *volumeMin;
@property (assign, nonatomic) IBOutlet NSTextField *volumeMax;
@property (assign, nonatomic) IBOutlet NSTextField *volumeVal;

@property (assign, nonatomic) IBOutlet NSSlider *radiusSlider;
@property (assign, nonatomic) IBOutlet NSTextField *radiusMin;
@property (assign, nonatomic) IBOutlet NSTextField *radiusMax;
@property (assign, nonatomic) IBOutlet NSTextField *radiusVal;

@property (assign, nonatomic) IBOutlet NSTextField *shortCutoff;
@property (assign, nonatomic) IBOutlet NSTextField *longCutoff;

-(IBAction)run:(id)sender;

-(IBAction)step:(id)sender;

-(IBAction)stop:(id)sender;

-(IBAction)reset:(id)sender;

-(IBAction)valueChanged:(id)sender;



@end
